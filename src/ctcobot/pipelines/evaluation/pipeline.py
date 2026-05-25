from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    run_eval_pipeline,
    compute_retrieval_metrics,
    compute_quality_metrics,
    run_latency_eval,
    save_benchmark_report,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=run_eval_pipeline,
            inputs=[
                "eval_qa_pairs",
                "params:ollama_base_url",
                "params:embedding_model",
                "params:llm_model",
                "params:chroma_persist_path",
                "params:chroma_collection_name",
                "params:top_k",
                "params:reranker_model",
                "params:rerank_top_n",
            ],
            outputs="qa_eval_results",
            name="run_eval_pipeline_node",
        ),
        node(
            func=compute_retrieval_metrics,
            inputs=[
                "qa_eval_results",
                "params:chroma_persist_path",
                "params:chroma_collection_name",
                "params:top_k",
            ],
            outputs="retrieval_results",
            name="compute_retrieval_metrics_node",
        ),
        node(
            func=compute_quality_metrics,
            inputs=[
                "qa_eval_results",
                "params:ollama_base_url",
                "params:judge_model",
            ],
            outputs="quality_results",
            name="compute_quality_metrics_node",
        ),
        node(
            func=run_latency_eval,
            inputs=[
                "params:ollama_base_url",
                "params:embedding_model",
                "params:llm_model",
                "params:chroma_persist_path",
                "params:chroma_collection_name",
                "params:top_k",
                "params:reranker_model",
                "params:rerank_top_n",
                "params:eval_latency_questions",
                "params:eval_num_latency_runs",
            ],
            outputs="latency_results",
            name="run_latency_eval_node",
        ),
        node(
            func=save_benchmark_report,
            inputs=[
                "retrieval_results",
                "quality_results",
                "latency_results",
            ],
            outputs="benchmark_report",
            name="save_benchmark_report_node",
        ),
    ])
