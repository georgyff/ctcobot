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

from ctcobot.pipelines.querying.nodes import build_prompt, generate_answer
from ctcobot.pipelines.querying.tools import VectorRAGTool
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
) -> list[dict]:
    """
    Run the HyDE vector RAG pipeline once per QA pair.

    Uses VectorRAGTool directly (HyDE → embed → turbovec → rerank → generate).
    Captures pre-rerank sources for retrieval metrics and the generated answer
    for quality metrics.
    """
    tool = VectorRAGTool(
        ollama_base_url=ollama_base_url,
        embedding_model=embedding_model,
        llm_model=llm_model,
        turbovec_persist_path=turbovec_persist_path,
        top_k=top_k,
        reranker_model=reranker_model,
        rerank_top_n=rerank_top_n,
    )

    results = []

    for _, row in eval_qa_pairs.iterrows():
        question = row["question"]
        logger.info("Running pipeline for: %s", question[:60])

        try:
            chunks = tool(question)
            prompt_data = build_prompt(question, chunks)
            result = generate_answer(prompt_data, ollama_base_url, llm_model)
            results.append({
                "question": question,
                "expected_source": row["source_file"],
                "expected_answer": row["expected_answer"],
                "pre_rerank_sources": tool.last_pre_rerank_sources,
                "answer": result["answer"],
                "sources": result["sources"],
                "tools_used": ["vector_rag"],
                "error": None,
            })
        except Exception as e:
            logger.warning("Pipeline failed for: %s — %s", question[:50], e)
            results.append({
                "question": question,
                "expected_source": row["source_file"],
                "expected_answer": row["expected_answer"],
                "pre_rerank_sources": [],
                "answer": "ERROR",
                "sources": [],
                "tools_used": [],
                "error": str(e),
            })

    logger.info("Eval pipeline complete: %d questions processed.", len(results))
    return results


def compute_retrieval_metrics(
    qa_eval_results: list[dict],
    turbovec_persist_path: str,
    top_k: int,
) -> dict:
    """
    Compute retrieval metrics (hit rate, MRR, precision, recall) from pre-rerank chunk data.
    """
    meta_path = Path(turbovec_persist_path) / "meta.json"
    with open(meta_path) as f:
        meta_doc = json.load(f)

    source_chunk_counts: dict[str, int] = {}
    for m in meta_doc["chunks"]:
        src = m.get("source_path", "")
        source_chunk_counts[src] = source_chunk_counts.get(src, 0) + 1

    results = []
    hits = 0
    reciprocal_ranks = []

    for item in qa_eval_results:
        expected_source = item["expected_source"]
        retrieved_sources = item["pre_rerank_sources"]

        hit = False
        rank = 0
        for i, src in enumerate(retrieved_sources, start=1):
            if src.endswith(expected_source) or expected_source in src:
                hit = True
                rank = i
                break

        retrieved_relevant = sum(
            1 for src in retrieved_sources
            if src.endswith(expected_source) or expected_source in src
        )
        total_relevant = sum(
            count for src, count in source_chunk_counts.items()
            if src.endswith(expected_source) or expected_source in src
        )
        n_retrieved = len(retrieved_sources)
        precision = retrieved_relevant / n_retrieved if n_retrieved > 0 else 0.0
        recall = retrieved_relevant / total_relevant if total_relevant > 0 else 0.0

        hits += int(hit)
        reciprocal_ranks.append(1.0 / rank if rank > 0 else 0.0)
        results.append({
            "question": item["question"],
            "expected_source": expected_source,
            "retrieved_sources": retrieved_sources,
            "hit": hit,
            "rank": rank,
            "reciprocal_rank": 1.0 / rank if rank > 0 else 0.0,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
        })
        logger.info(
            "%s Rank=%s | %s",
            "✅" if hit else "❌",
            rank if hit else "-",
            item["question"][:60],
        )

    n = len(results)
    hit_rate = hits / n if n > 0 else 0.0
    mrr = sum(reciprocal_ranks) / n if n > 0 else 0.0
    avg_precision = sum(r["precision"] for r in results) / n if n > 0 else 0.0
    avg_recall = sum(r["recall"] for r in results) / n if n > 0 else 0.0

    logger.info(
        "Retrieval metrics: Hit Rate=%.3f | MRR=%.3f | P=%.3f | R=%.3f",
        hit_rate, mrr, avg_precision, avg_recall,
    )

    return {
        "hit_rate": round(hit_rate, 4),
        "mrr": round(mrr, 4),
        "avg_precision": round(avg_precision, 4),
        "avg_recall": round(avg_recall, 4),
        "hits": hits,
        "total": n,
        "top_k": top_k,
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


def run_latency_eval(
    ollama_base_url: str,
    embedding_model: str,
    llm_model: str,
    turbovec_persist_path: str,
    top_k: int,
    reranker_model: str,
    rerank_top_n: int,
    eval_latency_questions: list[str],
    eval_num_latency_runs: int,
) -> dict:
    """
    Measure end-to-end HyDE vector RAG pipeline latency.

    Times VectorRAGTool + build_prompt + generate_answer per question so latency
    includes HyDE generation, embedding, retrieval, reranking, and answer synthesis.

    Returns:
        Dict with p50, p95, p99 latencies and per-run timings.
    """
    tool = VectorRAGTool(
        ollama_base_url=ollama_base_url,
        embedding_model=embedding_model,
        llm_model=llm_model,
        turbovec_persist_path=turbovec_persist_path,
        top_k=top_k,
        reranker_model=reranker_model,
        rerank_top_n=rerank_top_n,
    )

    latencies = []
    errors = 0
    n_questions = len(eval_latency_questions)

    logger.info(
        "Starting latency eval: %d runs across %d questions...",
        eval_num_latency_runs, n_questions,
    )

    for i in range(eval_num_latency_runs):
        question = eval_latency_questions[i % n_questions]
        t_start = time.perf_counter()

        try:
            chunks = tool(question)
            prompt_data = build_prompt(question, chunks)
            generate_answer(prompt_data, ollama_base_url, llm_model)
            elapsed = time.perf_counter() - t_start
            latencies.append(elapsed)
            logger.info("Run %d/%d: %.2fs", i + 1, eval_num_latency_runs, elapsed)

        except Exception as e:
            logger.warning("Run %d failed: %s", i + 1, e)
            errors += 1

    latencies.sort()
    n = len(latencies)

    def percentile(data: list, p: int) -> float:
        if not data:
            return 0.0
        idx = max(0, int(len(data) * p / 100) - 1)
        return round(data[idx], 3)

    result = {
        "total_runs": eval_num_latency_runs,
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
        "Latency eval complete. P50=%.3fs | P95=%.3fs | P99=%.3fs",
        result["p50_seconds"], result["p95_seconds"], result["p99_seconds"],
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
        retrieval_results: Output of run_retrieval_eval.
        quality_results:   Output of run_quality_eval.
        latency_results:   Output of run_latency_eval.

    Returns:
        Aggregated benchmark report dict.
    """
    all_tools: list[str] = []
    for item in quality_results.get("details", []):
        all_tools.extend(item.get("tools_used", []))
    tools_used_distribution = {t: all_tools.count(t) for t in dict.fromkeys(all_tools)}

    report = {
        "benchmark_version": "4.2",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "targets": {
            "hit_rate_at_5": 0.70,
            "mrr": 0.55,
            "precision_at_k": 0.15,
            "recall_at_k": 0.35,
            "avg_quality_score": 3.5,
            "p95_latency_seconds": 5.0,
        },
        "results": {
            "retrieval": {
                "hit_rate_at_5": retrieval_results["hit_rate"],
                "mrr": retrieval_results["mrr"],
                "avg_precision": retrieval_results["avg_precision"],
                "avg_recall": retrieval_results["avg_recall"],
                "hits": retrieval_results["hits"],
                "total": retrieval_results["total"],
                "top_k": retrieval_results["top_k"],
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

    report_path = Path("data/08_reporting/benchmark_report_v4-2.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))

    r = report["results"]
    print("\n" + "=" * 60)
    print("  ctcobot BENCHMARK REPORT v4.2")
    print("=" * 60)
    print(f"  {'Metric':<30} {'Result':>8}  {'Target':>8}  {'Pass':>6}")
    print(f"  {'-'*30} {'-'*8}  {'-'*8}  {'-'*6}")
    print(
        f"  {'Hit Rate @ k':<30} "
        f"{r['retrieval']['hit_rate_at_5']:>8.3f}  "
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
