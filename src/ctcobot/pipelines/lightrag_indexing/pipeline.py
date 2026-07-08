from kedro.pipeline import Pipeline, node, pipeline

from .nodes import build_lightrag_index


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=build_lightrag_index,
            inputs=[
                "cleaned_docs",
                "params:lightrag",
                "params:ollama_base_url",
            ],
            outputs="lightrag_index_result",
            name="build_lightrag_index_node",
        ),
    ])
