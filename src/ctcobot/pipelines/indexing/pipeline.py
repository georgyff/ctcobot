from kedro.pipeline import Pipeline, node, pipeline

from .nodes import ingest_documents, pageindex_index_documents


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=ingest_documents,
            inputs=["params:raw_data_path"],
            outputs="raw_docs",
            name="ingest_documents_node",
        ),
        node(
            func=pageindex_index_documents,
            inputs=[
                "raw_docs",
                "params:raw_data_path",
                "params:pageindex_model",
                "params:pageindex_workspace",
                "params:ollama_base_url",
            ],
            outputs="doc_registry",
            name="pageindex_index_documents_node",
        ),
    ])
