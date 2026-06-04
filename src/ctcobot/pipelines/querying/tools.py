"""
RAG tool implementations for the querying pipeline.

Exposes a callable interface returning list[dict] with keys:
  {text, source_path, score, ...}
"""
import logging

logger = logging.getLogger(__name__)


class VectorRAGTool:
    """HyDE + turbovec cosine-similarity retrieval with LLM listwise reranking."""

    def __init__(
        self,
        ollama_base_url: str,
        embedding_model: str,
        llm_model: str,
        turbovec_persist_path: str,
        top_k: int,
        reranker_model: str,
        rerank_top_n: int,
    ):
        self.ollama_base_url = ollama_base_url
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.turbovec_persist_path = turbovec_persist_path
        self.top_k = top_k
        self.reranker_model = reranker_model
        self.rerank_top_n = rerank_top_n
        self.last_pre_rerank_sources: list[str] = []

    def __call__(self, query: str) -> list[dict]:
        from ctcobot.pipelines.querying.nodes import (
            generate_hyde_doc,
            embed_query,
            retrieve_chunks,
            rerank_chunks,
        )
        hyde = generate_hyde_doc(query, self.ollama_base_url, self.llm_model)
        emb = embed_query(hyde, self.ollama_base_url, self.embedding_model)
        raw = retrieve_chunks(emb, self.turbovec_persist_path, self.top_k)
        self.last_pre_rerank_sources = [c["source_path"] for c in raw]
        return rerank_chunks(
            raw, query, self.reranker_model, self.rerank_top_n, self.ollama_base_url
        )
