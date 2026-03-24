from kedro.pipeline import Pipeline, node, pipeline

from .nodes import embed_query, retrieve_chunks, build_prompt, generate_answer


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=embed_query,
            inputs=[
                "params:question",
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