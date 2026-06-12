"""
RAG tool implementations for the querying pipeline.

Exposes a callable interface returning list[dict] with keys:
  {text, source_path, score, ...}
"""
import logging

logger = logging.getLogger(__name__)


class VectorRAGTool:
    """HyDE + folder-priority turbovec retrieval with LLM listwise reranking.

    Pipeline:
      1. generate_hyde_doc   — LLM writes a hypothetical policy paragraph
      2. embed_query         — embed the HyDE paragraph
      3. rank_folders        — LLM ranks all handbook folders by relevance
      4. retrieve_chunks_folder_priority
                             — restrict retrieval to the top-N ranked folders
      5. rerank_chunks       — LLM listwise rerank of candidate chunks
    """

    def __init__(
        self,
        ollama_base_url: str,
        embedding_model: str,
        llm_model: str,
        turbovec_persist_path: str,
        top_k: int,
        reranker_model: str,
        rerank_top_n: int,
        top_folders: int,
        retrieve_oversample: int,
    ):
        self.ollama_base_url = ollama_base_url
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.turbovec_persist_path = turbovec_persist_path
        self.top_k = top_k
        self.reranker_model = reranker_model
        self.rerank_top_n = rerank_top_n
        self.top_folders = top_folders
        self.retrieve_oversample = retrieve_oversample
        self.last_pre_rerank_sources: list[str] = []
        self.last_ranked_folders: list[str] = []

    def __call__(self, query: str) -> list[dict]:
        from ctcobot.pipelines.querying.nodes import (
            generate_hyde_doc,
            embed_query,
            rank_folders,
            retrieve_chunks_folder_priority,
            rerank_chunks,
        )
        hyde = generate_hyde_doc(query, self.ollama_base_url, self.llm_model)
        emb = embed_query(hyde, self.ollama_base_url, self.embedding_model)
        ranked = rank_folders(
            query, self.turbovec_persist_path, self.ollama_base_url, self.llm_model,
        )
        self.last_ranked_folders = ranked
        raw = retrieve_chunks_folder_priority(
            emb,
            ranked,
            self.turbovec_persist_path,
            self.top_k,
            self.top_folders,
            self.retrieve_oversample,
        )
        self.last_pre_rerank_sources = [c["source_path"] for c in raw]
        return rerank_chunks(
            raw, query, self.reranker_model, self.rerank_top_n, self.ollama_base_url
        )
