from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    run_retrieval_eval,
    run_quality_eval,
    run_latency_eval,
    save_benchmark_report,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=run_retrieval_eval,
            inputs=[
                "eval_qa_pairs",
                "params:ollama_base_url",
                "params:embedding_model",
                "params:llm_model",
                "params:chroma_persist_path",
                "params:chroma_collection_name",
                "params:top_k",
            ],
            outputs="retrieval_results",
            name="run_retrieval_eval_node",
        ),
        node(
            func=run_quality_eval,
            inputs=[
                "eval_qa_pairs",
                "params:ollama_base_url",
                "params:embedding_model",
                "params:llm_model",
                "params:judge_model",
                "params:chroma_persist_path",
                "params:chroma_collection_name",
                "params:top_k",
                "params:reranker_model",
                "params:rerank_top_n",
            ],
            outputs="quality_results",
            name="run_quality_eval_node",
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
