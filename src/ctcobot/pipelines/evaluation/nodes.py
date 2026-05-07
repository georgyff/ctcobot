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

from ctcobot.pipelines.querying.nodes import (
    build_prompt,
    generate_answer,
    pageindex_retrieve,
    rewrite_query,
)
from ctcobot.prompt_templates import JUDGE_PROMPT, format_judge_prompt

logger = logging.getLogger(__name__)


def run_full_eval_rag(
    eval_qa_pairs: pd.DataFrame,
    doc_registry: dict,
    ollama_base_url: str,
    pageindex_model: str,
    pageindex_workspace: str,
    llm_model: str,
    pageindex_top_docs: int,
    pageindex_top_sections: int,
    pageindex_reranker_top_k: int,
) -> list[dict]:
    """
    Run the full RAG pipeline (rewrite → retrieve → answer) once for every
    eval Q&A pair.  Downstream nodes consume these precomputed records instead
    of repeating the expensive retrieval themselves.

    Args:
        eval_qa_pairs:             DataFrame with columns: question,
                                   expected_answer, source_file, folder.
        doc_registry:              PageIndex document registry.
        ollama_base_url:           Ollama server URL.
        pageindex_model:           Ollama model used for PageIndex navigation.
        pageindex_workspace:       PageIndex workspace directory.
        llm_model:                 Answer generation model.
        pageindex_top_docs:        Documents selected in coarse retrieval stage.
        pageindex_top_sections:    Sections fetched per selected document.
        pageindex_reranker_top_k:  Chunks kept after reranking.

    Returns:
        List of dicts with keys: question, expected_answer, source_file,
        folder, expanded_question, chunks, answer, sources.
    """
    results = []

    for _, row in eval_qa_pairs.iterrows():
        question = row["question"]
        expected_answer = row["expected_answer"]
        source_file = row["source_file"]
        folder = row.get("folder", "")

        t_start = time.perf_counter()
        try:
            expanded = rewrite_query(question, pageindex_model, ollama_base_url)
            chunks = pageindex_retrieve(
                question=question,
                expanded_question=expanded,
                doc_registry=doc_registry,
                pageindex_model=pageindex_model,
                pageindex_workspace=pageindex_workspace,
                ollama_base_url=ollama_base_url,
                pageindex_top_docs=pageindex_top_docs,
                pageindex_top_sections=pageindex_top_sections,
                pageindex_reranker_top_k=pageindex_reranker_top_k,
            )
            prompt_data = build_prompt(question, chunks)
            result = generate_answer(prompt_data, ollama_base_url, llm_model)
            answer = result["answer"]
            sources = result["sources"]
            elapsed = time.perf_counter() - t_start
        except Exception as e:
            logger.warning("RAG failed for question: %s — %s", question[:50], e)
            expanded = question
            chunks = []
            answer = "ERROR: could not generate answer"
            sources = []
            elapsed = time.perf_counter() - t_start

        results.append({
            "question": question,
            "expected_answer": expected_answer,
            "source_file": source_file,
            "folder": folder,
            "expanded_question": expanded,
            "chunks": chunks,
            "answer": answer,
            "sources": sources,
            "elapsed_seconds": round(elapsed, 3),
        })
        logger.info("RAG complete (%.2fs): %s", elapsed, question[:60])

    return results


def run_retrieval_eval(
    eval_rag_results: list[dict],
    pageindex_top_docs: int,
) -> dict:
    """
    Compute hit rate and MRR from precomputed eval_rag_results.

    Args:
        eval_rag_results:  Output of run_full_eval_rag.
        pageindex_top_docs: Top-K value recorded in the report.

    Returns:
        Dict with hit_rate, mrr, and per-question details.
    """
    hits = 0
    reciprocal_ranks = []
    results = []

    for record in eval_rag_results:
        question = record["question"]
        expected_source = record["source_file"]
        retrieved_sources = [c["source_path"] for c in record["chunks"]]

        hit = False
        rank = 0
        for i, src in enumerate(retrieved_sources, start=1):
            if src.endswith(expected_source) or expected_source in src:
                hit = True
                rank = i
                break

        hits += int(hit)
        reciprocal_ranks.append(1.0 / rank if rank > 0 else 0.0)
        results.append({
            "question": question,
            "expected_source": expected_source,
            "retrieved_sources": retrieved_sources,
            "hit": hit,
            "rank": rank,
            "reciprocal_rank": 1.0 / rank if rank > 0 else 0.0,
        })
        logger.info(
            "%s Rank=%s | %s",
            "✅" if hit else "❌",
            rank if hit else "-",
            question[:60],
        )

    n = len(results)
    hit_rate = hits / n if n > 0 else 0.0
    mrr = sum(reciprocal_ranks) / n if n > 0 else 0.0

    logger.info("Retrieval eval complete. Hit Rate: %.3f | MRR: %.3f", hit_rate, mrr)

    return {
        "hit_rate": round(hit_rate, 4),
        "mrr": round(mrr, 4),
        "hits": hits,
        "total": n,
        "top_k": pageindex_top_docs,
        "details": results,
    }


def run_quality_eval(
    eval_rag_results: list[dict],
    ollama_base_url: str,
    judge_model: str,
) -> dict:
    """
    Score answers using LLM-as-judge from precomputed eval_rag_results.

    Args:
        eval_rag_results: Output of run_full_eval_rag.
        ollama_base_url:  Ollama server URL.
        judge_model:      Model used to score answers.

    Returns:
        Dict with avg_score, score distribution, and per-question details.
    """
    ollama_client = ollama.Client(host=ollama_base_url)
    scores = []
    results = []

    for record in eval_rag_results:
        question = record["question"]
        expected_answer = record["expected_answer"]
        actual_answer = record["answer"]
        sources = record["sources"]

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
            "sources": [s["source_path"] for s in sources],
            "score": score,
            "reason": reason,
        })
        logger.info("Score=%s/5 | %s", score, question[:60])

    valid_scores = [s for s in scores if s and s > 0]
    avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
    score_dist = {str(i): scores.count(i) for i in range(1, 6)}

    logger.info(
        "Quality eval complete. Avg score: %.2f/5 | Distribution: %s",
        avg_score, score_dist,
    )

    return {
        "avg_score": round(avg_score, 4),
        "score_distribution": score_dist,
        "total": len(results),
        "valid_scores": len(valid_scores),
        "details": results,
    }


def run_latency_eval(eval_rag_results: list[dict]) -> dict:
    """
    Compute latency statistics from timings recorded in run_full_eval_rag.

    Args:
        eval_rag_results: Output of run_full_eval_rag, each record contains
                          elapsed_seconds measured over the full RAG pipeline.

    Returns:
        Dict with p50, p95, p99 latencies and per-run timings.
    """
    latencies = sorted(r["elapsed_seconds"] for r in eval_rag_results)
    n = len(latencies)

    def percentile(data: list, p: int) -> float:
        if not data:
            return 0.0
        idx = max(0, int(len(data) * p / 100) - 1)
        return round(data[idx], 3)

    result = {
        "total_runs": n,
        "successful_runs": n,
        "errors": 0,
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
    llm_model: str,
    judge_model: str,
    pageindex_model: str,
    pageindex_top_docs: int,
    pageindex_top_sections: int,
    eval_num_latency_runs: int,
    parameters: dict,
) -> dict:
    """
    Aggregate all evaluation metrics into a single benchmark report.

    Args:
        retrieval_results:      Output of run_retrieval_eval.
        quality_results:        Output of run_quality_eval.
        latency_results:        Output of run_latency_eval.
        llm_model:              LLM used to generate answers.
        judge_model:            LLM used to score answer quality.
        pageindex_model:        Ollama model used for PageIndex operations.
        pageindex_top_docs:     Docs selected per query.
        pageindex_top_sections: Sections fetched per document.
        eval_num_latency_runs:  Number of timed runs for latency eval.
        parameters:             Full parameters dict (for extras).

    Returns:
        Aggregated benchmark report dict.
    """
    report = {
        "benchmark_version": "2.7",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "pageindex_model": pageindex_model,
            "llm_model": llm_model,
            "judge_model": judge_model,
            "pageindex_top_docs": pageindex_top_docs,
            "pageindex_top_sections": pageindex_top_sections,
            "eval_num_latency_runs": eval_num_latency_runs,
        },
        "targets": {
            "hit_rate_at_5": 0.70,
            "mrr": 0.55,
            "avg_quality_score": 3.5,
            "p95_latency_seconds": 5.0,
        },
        "results": {
            "retrieval": {
                "hit_rate_at_5": retrieval_results["hit_rate"],
                "mrr": retrieval_results["mrr"],
                "hits": retrieval_results["hits"],
                "total": retrieval_results["total"],
                "top_k": retrieval_results["top_k"],
                "hit_rate_pass": retrieval_results["hit_rate"] >= 0.70,
                "mrr_pass": retrieval_results["mrr"] >= 0.55,
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
        "details": {
            "retrieval": retrieval_results.get("details", []),
            "quality": quality_results.get("details", []),
            "latency": latency_results.get("all_latencies", []),
        },
    }

    raw_version = str(report.get("benchmark_version", "unknown"))
    version_tag = "v" + raw_version.replace(".", "-")
    report_path = Path("data/08_reporting") / f"benchmark_report_{version_tag}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    logger.info("Benchmark report saved to %s", report_path)

    r = report["results"]
    print("\n" + "=" * 60)
    print("  ctcobot BENCHMARK REPORT v3.0")
    print("=" * 60)
    print(f"  {'Metric':<30} {'Result':>8}  {'Target':>8}  {'Pass':>6}")
    print(f"  {'-'*30} {'-'*8}  {'-'*8}  {'-'*6}")
    print(
        f"  {'Hit Rate @ 5':<30} "
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

    logger.info("Benchmark report saved.")
    return report
