"""
Evaluation pipeline nodes — retrieval, quality, latency, reporting.
"""
import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import ollama
import pandas as pd

from ctcobot.pipelines.querying.agent import QueryAgent
from ctcobot.pipelines.querying.tools import KeywordRAGTool, VectorRAGTool
from ctcobot.prompt_templates import JUDGE_PROMPT, format_judge_prompt

logger = logging.getLogger(__name__)


def run_eval_pipeline(
    eval_qa_pairs: pd.DataFrame,
    ollama_base_url: str,
    embedding_model: str,
    llm_model: str,
    turbovec_persist_path: str,
    top_k: int,
    reranker_model: str,
    rerank_top_n: int,
    top_folders: int,
    retrieve_oversample: int,
    agent_model: str,
    bm25_top_k: int,
    priority_folders: list[str],
    agent_reranker_model: str,
) -> list[dict]:
    """
    Run the agentic RAG pipeline once per QA pair.

    A QueryAgent always runs the vector HyDE tool and, when the router or the
    acronym heuristic calls for it, also the folder-scoped BM25 keyword tool;
    it merges + reranks the results and generates the answer. Both tools share
    a guaranteed HR-folder floor (``priority_folders``). Captures the union of
    pre-rerank sources (for retrieval metrics), which tool(s) were used, and
    the answer (for quality).
    """
    vector_tool = VectorRAGTool(
        ollama_base_url=ollama_base_url,
        embedding_model=embedding_model,
        llm_model=llm_model,
        turbovec_persist_path=turbovec_persist_path,
        top_k=top_k,
        reranker_model=reranker_model,
        rerank_top_n=rerank_top_n,
        top_folders=top_folders,
        retrieve_oversample=retrieve_oversample,
        priority_folders=priority_folders,
    )
    keyword_tool = KeywordRAGTool(
        ollama_base_url=ollama_base_url,
        llm_model=llm_model,
        turbovec_persist_path=turbovec_persist_path,
        top_k=bm25_top_k,
        top_folders=top_folders,
        priority_folders=priority_folders,
    )
    agent = QueryAgent(
        tools={"vector_rag": vector_tool, "keyword_rag": keyword_tool},
        ollama_base_url=ollama_base_url,
        agent_model=agent_model,
        llm_model=llm_model,
        reranker_model=reranker_model,
        rerank_top_n=rerank_top_n,
        turbovec_persist_path=turbovec_persist_path,
        agent_reranker_model=agent_reranker_model,
    )

    results = []

    for _, row in eval_qa_pairs.iterrows():
        question = row["question"]
        logger.info("Running pipeline for: %s", question[:60])

        t_start = time.perf_counter()
        try:
            result = agent.run(question)
            elapsed = time.perf_counter() - t_start
            logger.info(
                "Pipeline run: %.2fs | tools=%s | %s",
                elapsed, result["tools_used"], question[:60],
            )
            results.append({
                "question": question,
                "expected_source": row["source_file"],
                "expected_answer": row["expected_answer"],
                "pre_rerank_sources": result["pre_rerank_sources"],
                "pre_rerank_chunks": result.get("pre_rerank_chunks", []),
                "answer": result["answer"],
                "sources": result["sources"],
                "tools_used": result["tools_used"],
                "latency_seconds": round(elapsed, 3),
                "error": None,
            })
        except Exception as e:
            elapsed = time.perf_counter() - t_start
            logger.warning("Pipeline failed for: %s — %s", question[:50], e)
            results.append({
                "question": question,
                "expected_source": row["source_file"],
                "expected_answer": row["expected_answer"],
                "pre_rerank_sources": [],
                "pre_rerank_chunks": [],
                "answer": "ERROR",
                "sources": [],
                "tools_used": [],
                "latency_seconds": None,
                "error": str(e),
            })

    logger.info("Eval pipeline complete: %d questions processed.", len(results))
    return results


_NO_RETRIEVAL_TARGET = {"", "-", "none", "n/a", "na"}


def _source_matches(retrieved: str, expected: str) -> bool:
    """True if a retrieved source path refers to the expected source doc."""
    return retrieved.endswith(expected) or expected in retrieved


def compute_retrieval_metrics(
    qa_eval_results: list[dict],
    turbovec_persist_path: str,
    retrieval_eval_k: int,
) -> dict:
    """
    Compute retrieval metrics over a fixed, deduplicated top-k candidate window.

    Candidates come from ``pre_rerank_chunks`` (deduplicated by
    ``(source_path, chunk_index)`` in the agent), truncated to the first
    ``retrieval_eval_k`` for every question so precision/recall are comparable
    regardless of how many tools ran. Recall is bounded by ``min(total_relevant,
    retrieval_eval_k)`` so it is reachable and never exceeds 1.0.

    Out-of-scope questions (expected_source blank or a no-target sentinel, e.g.
    the refusal case) are excluded from retrieval aggregates — they are answer-
    quality cases, not retrieval cases.
    """
    meta_path = Path(turbovec_persist_path) / "meta.json"
    with open(meta_path) as f:
        meta_doc = json.load(f)

    source_chunk_counts: dict[str, int] = {}
    for m in meta_doc["chunks"]:
        src = m.get("source_path", "")
        source_chunk_counts[src] = source_chunk_counts.get(src, 0) + 1

    results = []
    scored = 0           # questions with a real retrieval target
    hits = 0
    reciprocal_ranks: list[float] = []
    precisions: list[float] = []
    recalls: list[float] = []

    for item in qa_eval_results:
        expected_source = (item.get("expected_source") or "").strip()

        # Prefer deduped (source, chunk) candidates; fall back to source list.
        chunks = item.get("pre_rerank_chunks") or [
            {"source_path": s, "chunk_index": None}
            for s in item.get("pre_rerank_sources", [])
        ]
        topk = chunks[:retrieval_eval_k]
        topk_sources = [c["source_path"] for c in topk]

        if expected_source.lower() in _NO_RETRIEVAL_TARGET:
            results.append({
                "question": item["question"],
                "expected_source": expected_source,
                "retrieved_sources": topk_sources,
                "hit": None,
                "rank": 0,
                "reciprocal_rank": 0.0,
                "precision": None,
                "recall": None,
                "retrieval_scored": False,
            })
            continue

        hit = False
        rank = 0
        for i, src in enumerate(topk_sources, start=1):
            if _source_matches(src, expected_source):
                hit, rank = True, i
                break

        relevant_in_topk = sum(
            1 for src in topk_sources if _source_matches(src, expected_source)
        )
        total_relevant = sum(
            count for src, count in source_chunk_counts.items()
            if _source_matches(src, expected_source)
        )
        k = len(topk) if topk else 1
        precision = relevant_in_topk / k
        recall_denom = min(total_relevant, retrieval_eval_k) or 1
        recall = relevant_in_topk / recall_denom

        scored += 1
        hits += int(hit)
        rr = 1.0 / rank if rank > 0 else 0.0
        reciprocal_ranks.append(rr)
        precisions.append(precision)
        recalls.append(recall)
        results.append({
            "question": item["question"],
            "expected_source": expected_source,
            "retrieved_sources": topk_sources,
            "hit": hit,
            "rank": rank,
            "reciprocal_rank": rr,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "retrieval_scored": True,
        })
        logger.info(
            "%s Rank=%s | %s",
            "✅" if hit else "❌",
            rank if hit else "-",
            item["question"][:60],
        )

    hit_rate = hits / scored if scored else 0.0
    mrr = sum(reciprocal_ranks) / scored if scored else 0.0
    avg_precision = sum(precisions) / scored if scored else 0.0
    avg_recall = sum(recalls) / scored if scored else 0.0

    logger.info(
        "Retrieval metrics @k=%d (n=%d): Hit Rate=%.3f | MRR=%.3f | P=%.3f | R=%.3f",
        retrieval_eval_k, scored, hit_rate, mrr, avg_precision, avg_recall,
    )

    return {
        "hit_rate": round(hit_rate, 4),
        "mrr": round(mrr, 4),
        "avg_precision": round(avg_precision, 4),
        "avg_recall": round(avg_recall, 4),
        "hits": hits,
        "total": scored,
        "retrieval_eval_k": retrieval_eval_k,
        "details": results,
    }


def compute_quality_metrics(
    qa_eval_results: list[dict],
    ollama_base_url: str,
    judge_model: str,
) -> dict:
    """
    Score answer quality using an LLM judge on pre-generated answers.
    """
    ollama_client = ollama.Client(host=ollama_base_url)
    results = []
    scores = []

    for item in qa_eval_results:
        question = item["question"]
        expected_answer = item["expected_answer"]
        actual_answer = item["answer"]

        if item["error"]:
            scores.append(0)
            results.append({
                "question": question,
                "expected_answer": expected_answer,
                "actual_answer": actual_answer,
                "sources": [],
                "score": 0,
                "reason": f"pipeline error: {item['error']}",
            })
            continue

        judge_user_prompt = format_judge_prompt(
            question=question,
            expected_answer=expected_answer,
            actual_answer=actual_answer,
        )

        score = None
        reason = "parse error"
        try:
            judge_response = ollama_client.chat(
                model=judge_model,
                messages=[
                    {"role": "system", "content": JUDGE_PROMPT},
                    {"role": "user", "content": judge_user_prompt},
                ],
                think=False,
                # Deterministic scoring — same answer gets the same score,
                # the single biggest lever for benchmark reproducibility.
                options={"temperature": 0.0},
            )
            raw = judge_response["message"]["content"].strip()

            match = re.search(r'\{.*?\}', raw, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
                score = int(parsed["score"])
                reason = parsed.get("reason", "")
            else:
                logger.warning("Could not parse judge response: %s", raw[:100])
                score = 0

        except Exception as e:
            logger.warning("Judge failed for question: %s — %s", question[:50], e)
            score = 0

        scores.append(score)
        results.append({
            "question": question,
            "expected_answer": expected_answer,
            "actual_answer": actual_answer,
            "sources": [s["source_path"] for s in item["sources"]],
            "tools_used": item.get("tools_used", []),
            "score": score,
            "reason": reason,
        })
        logger.info("Score=%s/5 | %s", score, question[:60])

    valid_scores = [s for s in scores if s and s > 0]
    avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
    score_dist = {str(i): scores.count(i) for i in range(1, 6)}

    logger.info(
        "Quality metrics: Avg=%.2f/5 | Distribution=%s",
        avg_score, score_dist,
    )

    return {
        "avg_score": round(avg_score, 4),
        "score_distribution": score_dist,
        "total": len(results),
        "valid_scores": len(valid_scores),
        "details": results,
    }


def compute_latency_metrics(qa_eval_results: list[dict]) -> dict:
    """
    Compute latency percentiles from per-question timings recorded during
    ``run_eval_pipeline``.

    No separate latency loop is run — each of the 25 eval questions is timed
    end-to-end (HyDE → folder rank → retrieve → rerank → answer) during the
    main pipeline pass, and this node aggregates those timings.

    Args:
        qa_eval_results: Output of ``run_eval_pipeline``. Each entry must
                         carry a ``latency_seconds`` field (``None`` on error).

    Returns:
        Dict with p50, p95, p99, min, max, avg, and the sorted latency list.
        Shape matches what ``save_benchmark_report`` expects.
    """
    latencies = sorted(
        r["latency_seconds"]
        for r in qa_eval_results
        if r.get("latency_seconds") is not None
    )
    n = len(latencies)
    total = len(qa_eval_results)
    errors = total - n

    def percentile(data: list, p: int) -> float:
        if not data:
            return 0.0
        idx = max(0, int(len(data) * p / 100) - 1)
        return round(data[idx], 3)

    result = {
        "total_runs": total,
        "successful_runs": n,
        "errors": errors,
        "p50_seconds": percentile(latencies, 50),
        "p95_seconds": percentile(latencies, 95),
        "p99_seconds": percentile(latencies, 99),
        "min_seconds": round(min(latencies), 3) if latencies else 0,
        "max_seconds": round(max(latencies), 3) if latencies else 0,
        "avg_seconds": round(sum(latencies) / n, 3) if latencies else 0,
        "all_latencies": latencies,
    }

    logger.info(
        "Latency from eval run (n=%d): P50=%.3fs | P95=%.3fs | P99=%.3fs",
        n, result["p50_seconds"], result["p95_seconds"], result["p99_seconds"],
    )
    return result


def save_benchmark_report(
    retrieval_results: dict,
    quality_results: dict,
    latency_results: dict,
) -> dict:
    """
    Aggregate all evaluation metrics into a single benchmark report.

    Args:
        retrieval_results: Output of compute_retrieval_metrics.
        quality_results:   Output of compute_quality_metrics.
        latency_results:   Output of compute_latency_metrics
                           (derived from per-question timings in qa_eval_results).

    Returns:
        Aggregated benchmark report dict.
    """
    all_tools: list[str] = []
    for item in quality_results.get("details", []):
        all_tools.extend(item.get("tools_used", []))
    tools_used_distribution = {t: all_tools.count(t) for t in dict.fromkeys(all_tools)}

    report = {
        "benchmark_version": "5.4",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "targets": {
            "hit_rate_at_k": 0.70,
            "mrr": 0.55,
            "precision_at_k": 0.15,
            "recall_at_k": 0.35,
            "avg_quality_score": 3.5,
            "p95_latency_seconds": 5.0,
        },
        "results": {
            "retrieval": {
                "hit_rate_at_k": retrieval_results["hit_rate"],
                "mrr": retrieval_results["mrr"],
                "avg_precision": retrieval_results["avg_precision"],
                "avg_recall": retrieval_results["avg_recall"],
                "hits": retrieval_results["hits"],
                "total": retrieval_results["total"],
                "retrieval_eval_k": retrieval_results["retrieval_eval_k"],
                "hit_rate_pass": retrieval_results["hit_rate"] >= 0.70,
                "mrr_pass": retrieval_results["mrr"] >= 0.55,
                "precision_pass": retrieval_results["avg_precision"] >= 0.15,
                "recall_pass": retrieval_results["avg_recall"] >= 0.35,
            },
            "quality": {
                "avg_score": quality_results["avg_score"],
                "score_distribution": quality_results["score_distribution"],
                "total": quality_results["total"],
                "valid_scores": quality_results["valid_scores"],
                "quality_pass": quality_results["avg_score"] >= 3.5,
            },
            "latency": {
                "p50_seconds": latency_results["p50_seconds"],
                "p95_seconds": latency_results["p95_seconds"],
                "p99_seconds": latency_results["p99_seconds"],
                "avg_seconds": latency_results["avg_seconds"],
                "successful_runs": latency_results["successful_runs"],
                "total_runs": latency_results["total_runs"],
                "latency_pass": latency_results["p95_seconds"] <= 5.0,
            },
        },
        "tools_used_distribution": tools_used_distribution,
        "details": {
            "retrieval": retrieval_results.get("details", []),
            "quality": quality_results.get("details", []),
            "latency": latency_results.get("all_latencies", []),
        },
    }

    report_path = Path("data/08_reporting/benchmark_report_v5-4.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))

    r = report["results"]
    print("\n" + "=" * 60)
    print("  ctcobot BENCHMARK REPORT v5.4")
    print("=" * 60)
    print(f"  {'Metric':<30} {'Result':>8}  {'Target':>8}  {'Pass':>6}")
    print(f"  {'-'*30} {'-'*8}  {'-'*8}  {'-'*6}")
    print(
        f"  {'Hit Rate @ k':<30} "
        f"{r['retrieval']['hit_rate_at_k']:>8.3f}  "
        f"{'≥ 0.700':>8}  "
        f"{'✅' if r['retrieval']['hit_rate_pass'] else '❌':>6}"
    )
    print(
        f"  {'MRR':<30} "
        f"{r['retrieval']['mrr']:>8.3f}  "
        f"{'≥ 0.550':>8}  "
        f"{'✅' if r['retrieval']['mrr_pass'] else '❌':>6}"
    )
    print(
        f"  {'Precision @ k':<30} "
        f"{r['retrieval']['avg_precision']:>8.3f}  "
        f"{'≥ 0.150':>8}  "
        f"{'✅' if r['retrieval']['precision_pass'] else '❌':>6}"
    )
    print(
        f"  {'Recall @ k':<30} "
        f"{r['retrieval']['avg_recall']:>8.3f}  "
        f"{'≥ 0.350':>8}  "
        f"{'✅' if r['retrieval']['recall_pass'] else '❌':>6}"
    )
    print(
        f"  {'Avg Quality Score (1-5)':<30} "
        f"{r['quality']['avg_score']:>8.3f}  "
        f"{'≥ 3.500':>8}  "
        f"{'✅' if r['quality']['quality_pass'] else '❌':>6}"
    )
    print(
        f"  {'P95 Latency (seconds)':<30} "
        f"{r['latency']['p95_seconds']:>8.3f}  "
        f"{'≤ 5.000':>8}  "
        f"{'✅' if r['latency']['latency_pass'] else '❌':>6}"
    )
    print("=" * 60 + "\n")

    logger.info("Benchmark report saved to %s", report_path)
    return report
