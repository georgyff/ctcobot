from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    rewrite_query,
    embed_query,
    retrieve_chunks,
    rerank_chunks,
    build_prompt,
    generate_answer,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=rewrite_query,
            inputs=[
                "params:question",
                "params:ollama_base_url",
                "params:llm_model",
            ],
            outputs="rewritten_query",
            name="rewrite_query_node",
        ),
        node(
            func=embed_query,
            inputs=[
                "rewritten_query",
                "params:ollama_base_url",
                "params:embedding_model",
            ],
            outputs="query_embedding",
            name="embed_query_node",
        ),
        node(
            func=retrieve_chunks,
            inputs=[
                "query_embedding",
                "params:chroma_persist_path",
                "params:chroma_collection_name",
                "params:top_k",
            ],
            outputs="retrieved_chunks",
            name="retrieve_chunks_node",
        ),
        node(
            func=rerank_chunks,
            inputs=[
                "retrieved_chunks",
                "params:question",
                "params:reranker_model",
                "params:rerank_top_n",
                "params:ollama_base_url",
            ],
            outputs="reranked_chunks",
            name="rerank_chunks_node",
        ),
        node(
            func=build_prompt,
            inputs=[
                "params:question",
                "reranked_chunks",
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
