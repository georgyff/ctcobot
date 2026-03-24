"""
Evaluation pipeline nodes — retrieval, quality, latency, reporting.
"""
import logging
import time
import ollama
import chromadb
import json
import re
import ollama

from ctcobot.pipelines.querying.nodes import (
        embed_query, retrieve_chunks, build_prompt, generate_answer
    )
from ctcobot.prompt_templates import JUDGE_PROMPT, format_judge_prompt

from ctcobot.pipelines.querying.nodes import (
        embed_query, retrieve_chunks, build_prompt, generate_answer
    )

from datetime import datetime

import pandas as pd

logger = logging.getLogger(__name__)


def run_retrieval_eval(
    eval_qa_pairs: pd.DataFrame,
    ollama_base_url: str,
    embedding_model: str,
    chroma_persist_path: str,
    chroma_collection_name: str,
    top_k: int,
) -> dict:
    """
    Evaluate retrieval quality using hit rate and MRR.

    For each Q&A pair, embed the question, retrieve top-k chunks,
    and check whether the expected source file appears in results.

    Args:
        eval_qa_pairs:          DataFrame with columns: question,
                                expected_answer, source_file, folder.
        ollama_base_url:        Ollama server URL.
        embedding_model:        Ollama embedding model name.
        chroma_persist_path:    ChromaDB path.
        chroma_collection_name: ChromaDB collection name.
        top_k:                  Number of chunks to retrieve.

    Returns:
        Dict with hit_rate, mrr, and per-question details.
    """

    ollama_client = ollama.Client(host=ollama_base_url)
    chroma_client = chromadb.PersistentClient(path=chroma_persist_path)
    collection = chroma_client.get_collection(name=chroma_collection_name)

    results = []
    hits = 0
    reciprocal_ranks = []

    for _, row in eval_qa_pairs.iterrows():
        question = row["question"]
        expected_source = row["source_file"]  # e.g. "people-group/anti-harassment.md"

        # Embed question
        response = ollama_client.embeddings(
            model=embedding_model,
            prompt=question,
        )
        embedding = response["embedding"]

        # Retrieve top-k
        query_results = collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["metadatas", "distances"],
        )

        retrieved_sources = [
            m.get("source_path", "")
            for m in query_results["metadatas"][0]
        ]

        # Check hit — expected source matches end of any retrieved source path
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
            f"{'✅' if hit else '❌'} "
            f"Rank={rank if hit else '-'} | {question[:60]}"
        )

    n = len(results)
    hit_rate = hits / n
    mrr = sum(reciprocal_ranks) / n

    logger.info(f"Retrieval eval complete. Hit Rate: {hit_rate:.3f} | MRR: {mrr:.3f}")

    return {
        "hit_rate": round(hit_rate, 4),
        "mrr": round(mrr, 4),
        "hits": hits,
        "total": n,
        "top_k": top_k,
        "details": results,
    }
    
def run_quality_eval(
    eval_qa_pairs: pd.DataFrame,
    ollama_base_url: str,
    embedding_model: str,
    llm_model: str,
    judge_model: str,
    chroma_persist_path: str,
    chroma_collection_name: str,
    top_k: int,
) -> dict:
    """
    Evaluate answer quality using an LLM-as-judge approach.

    For each Q&A pair, generate an answer with the full RAG pipeline,
    then score it 1-5 using the judge model.

    Args:
        eval_qa_pairs:          DataFrame with eval questions.
        ollama_base_url:        Ollama server URL.
        embedding_model:        Embedding model name.
        llm_model:              Answer generation model.
        judge_model:            Model used to score answers.
        chroma_persist_path:    ChromaDB path.
        chroma_collection_name: ChromaDB collection name.
        top_k:                  Chunks to retrieve per question.

    Returns:
        Dict with avg_score, score distribution, and per-question details.
    """

    ollama_client = ollama.Client(host=ollama_base_url)
    results = []
    scores = []

    for _, row in eval_qa_pairs.iterrows():
        question = row["question"]
        expected_answer = row["expected_answer"]

        # Generate answer via full RAG pipeline
        try:
            embedding = embed_query(question, ollama_base_url, embedding_model)
            chunks = retrieve_chunks(
                embedding, chroma_persist_path, chroma_collection_name, top_k
            )
            prompt_data = build_prompt(question, chunks)
            result = generate_answer(prompt_data, ollama_base_url, llm_model)
            actual_answer = result["answer"]
            sources = result["sources"]
        except Exception as e:
            logger.warning(f"RAG failed for question: {question[:50]} — {e}")
            actual_answer = "ERROR: could not generate answer"
            sources = []

        # Judge the answer
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
            )
            raw = judge_response["message"]["content"].strip()

            # Extract JSON robustly — find first { ... } block
            match = re.search(r'\{.*?\}', raw, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
                score = int(parsed["score"])
                reason = parsed.get("reason", "")
            else:
                logger.warning(f"Could not parse judge response: {raw[:100]}")
                score = 0

        except Exception as e:
            logger.warning(f"Judge failed for question: {question[:50]} — {e}")
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

        logger.info(f"Score={score}/5 | {question[:60]}")

    valid_scores = [s for s in scores if s and s > 0]
    avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

    score_dist = {str(i): scores.count(i) for i in range(1, 6)}

    logger.info(
        f"Quality eval complete. "
        f"Avg score: {avg_score:.2f}/5 | Distribution: {score_dist}"
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
    chroma_persist_path: str,
    chroma_collection_name: str,
    top_k: int,
    eval_latency_questions: list[str],
    eval_num_latency_runs: int,
) -> dict:
    """
    Measure end-to-end query pipeline latency.

    Runs the full pipeline eval_num_latency_runs times across the
    provided question set and computes P50, P95, P99 latencies.

    Args:
        ollama_base_url:          Ollama server URL.
        embedding_model:          Embedding model name.
        llm_model:                LLM model name.
        chroma_persist_path:      ChromaDB path.
        chroma_collection_name:   ChromaDB collection name.
        top_k:                    Chunks to retrieve.
        eval_latency_questions:   Fixed question set to cycle through.
        eval_num_latency_runs:    Total number of timed runs.

    Returns:
        Dict with p50, p95, p99 latencies and per-run timings.
    """

    latencies = []
    errors = 0
    n_questions = len(eval_latency_questions)

    logger.info(
        f"Starting latency eval: {eval_num_latency_runs} runs "
        f"across {n_questions} questions..."
    )

    for i in range(eval_num_latency_runs):
        question = eval_latency_questions[i % n_questions]
        t_start = time.perf_counter()

        try:
            embedding = embed_query(question, ollama_base_url, embedding_model)
            chunks = retrieve_chunks(
                embedding, chroma_persist_path, chroma_collection_name, top_k
            )
            prompt_data = build_prompt(question, chunks)
            generate_answer(prompt_data, ollama_base_url, llm_model)

            elapsed = time.perf_counter() - t_start
            latencies.append(elapsed)
            logger.info(f"Run {i+1}/{eval_num_latency_runs}: {elapsed:.2f}s")

        except Exception as e:
            logger.warning(f"Run {i+1} failed: {e}")
            errors += 1

    latencies.sort()
    n = len(latencies)

    def percentile(data, p):
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
        f"Latency eval complete. "
        f"P50={result['p50_seconds']}s | "
        f"P95={result['p95_seconds']}s | "
        f"P99={result['p99_seconds']}s"
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

    report = {
        "benchmark_version": "1.0",
        "timestamp": datetime.utcnow().isoformat(),
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

    # Print summary table to stdout
    r = report["results"]
    print("\n" + "=" * 60)
    print("  ctcobot BENCHMARK REPORT v1.0")
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