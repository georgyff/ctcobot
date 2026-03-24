from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    ingest_documents,
    clean_documents,
    chunk_documents,
    embed_and_index,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=ingest_documents,
            inputs=["params:raw_data_path"],
            outputs="raw_docs",
            name="ingest_documents_node",
        ),
        node(
            func=clean_documents,
            inputs=[
                "raw_docs",
                "params:min_doc_tokens",
                "params:encoding_name",
            ],
            outputs="cleaned_docs",
            name="clean_documents_node",
        ),
        node(
            func=chunk_documents,
            inputs=[
                "cleaned_docs",
                "params:chunk_size",
                "params:chunk_overlap",
                "params:encoding_name",
            ],
            outputs="chunked_docs",
            name="chunk_documents_node",
        ),
        node(
            func=embed_and_index,
            inputs=[
                "chunked_docs",
                "params:ollama_base_url",
                "params:embedding_model",
                "params:chroma_persist_path",
                "params:chroma_collection_name",
            ],
            outputs="indexing_summary",
            name="embed_and_index_node",
        ),
    ])