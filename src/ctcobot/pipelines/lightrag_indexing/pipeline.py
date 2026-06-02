from kedro.pipeline import Pipeline, node, pipeline

from .nodes import build_lightrag_index


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=build_lightrag_index,
            inputs=[
                "cleaned_docs",
                "params:lightrag_index_folders",
                "params:lightrag_working_dir",
                "params:ollama_base_url",
                "params:llm_model",
                "params:embedding_model",
            ],
            outputs="lightrag_index_result",
            name="build_lightrag_index_node",
        ),
    ])
