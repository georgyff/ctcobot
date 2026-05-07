from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    run_full_eval_rag,
    run_latency_eval,
    run_quality_eval,
    run_retrieval_eval,
    save_benchmark_report,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=run_full_eval_rag,
            inputs=[
                "eval_qa_pairs",
                "doc_registry",
                "params:ollama_base_url",
                "params:pageindex_model",
                "params:pageindex_workspace",
                "params:llm_model",
                "params:pageindex_top_docs",
                "params:pageindex_top_sections",
                "params:pageindex_reranker_top_k",
            ],
            outputs="eval_rag_results",
            name="run_full_eval_rag_node",
        ),
        node(
            func=run_retrieval_eval,
            inputs=[
                "eval_rag_results",
                "params:pageindex_top_docs",
            ],
            outputs="retrieval_results",
            name="run_retrieval_eval_node",
        ),
        node(
            func=run_quality_eval,
            inputs=[
                "eval_rag_results",
                "params:ollama_base_url",
                "params:judge_model",
            ],
            outputs="quality_results",
            name="run_quality_eval_node",
        ),
        node(
            func=run_latency_eval,
            inputs=["eval_rag_results"],
            outputs="latency_results",
            name="run_latency_eval_node",
        ),
        node(
            func=save_benchmark_report,
            inputs=[
                "retrieval_results",
                "quality_results",
                "latency_results",
                "params:llm_model",
                "params:judge_model",
                "params:pageindex_model",
                "params:pageindex_top_docs",
                "params:pageindex_top_sections",
                "params:eval_num_latency_runs",
                "parameters",
            ],
            outputs="benchmark_report",
            name="save_benchmark_report_node",
        ),
    ])
