from kedro.pipeline import Pipeline, node, pipeline

from .nodes import build_prompt, generate_answer, pageindex_retrieve, rewrite_query


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=rewrite_query,
            inputs=[
                "params:question",
                "params:pageindex_model",
                "params:ollama_base_url",
            ],
            outputs="expanded_question",
            name="rewrite_query_node",
        ),
        node(
            func=pageindex_retrieve,
            inputs=[
                "params:question",
                "expanded_question",
                "doc_registry",
                "params:pageindex_model",
                "params:pageindex_workspace",
                "params:ollama_base_url",
                "params:pageindex_top_docs",
                "params:pageindex_top_sections",
                "params:pageindex_reranker_top_k",
            ],
            outputs="retrieved_chunks",
            name="pageindex_retrieve_node",
        ),
        node(
            func=build_prompt,
            inputs=[
                "params:question",
                "retrieved_chunks",
            ],
            outputs="prompt_data",
            name="build_prompt_node",
        ),
        node(
            func=generate_answer,
            inputs=[
                "prompt_data",
                "params:ollama_base_url",
                "params:llm_model",
            ],
            outputs="query_results",
            name="generate_answer_node",
        ),
    ])
