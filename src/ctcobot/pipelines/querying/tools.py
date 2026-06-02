"""
RAG tool implementations for the v3.0 agentic query pipeline.

Each tool exposes a callable interface returning list[dict] with keys:
  {text, source_path, score, ...}
"""
import logging

logger = logging.getLogger(__name__)


class VectorRAGTool:
    """HyDE + ChromaDB cosine-similarity retrieval with LLM listwise reranking."""

    def __init__(
        self,
        ollama_base_url: str,
        embedding_model: str,
        llm_model: str,
        chroma_persist_path: str,
        chroma_collection_name: str,
        top_k: int,
        reranker_model: str,
        rerank_top_n: int,
    ):
        self.ollama_base_url = ollama_base_url
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.chroma_persist_path = chroma_persist_path
        self.chroma_collection_name = chroma_collection_name
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
        raw = retrieve_chunks(
            emb, self.chroma_persist_path, self.chroma_collection_name, self.top_k
        )
        self.last_pre_rerank_sources = [c["source_path"] for c in raw]
        return rerank_chunks(
            raw, query, self.reranker_model, self.rerank_top_n, self.ollama_base_url
        )


class GraphRAGTool:
    """LightRAG knowledge graph retrieval — entity/relationship aware search."""

    def __init__(
        self,
        lightrag_working_dir: str,
        ollama_base_url: str,
        llm_model: str,
        embedding_model: str,
    ):
        self.lightrag_working_dir = lightrag_working_dir
        self.ollama_base_url = ollama_base_url
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self._rag = None

    def _init_rag(self):
        import asyncio
        import numpy as np
        import ollama as _ollama
        from lightrag import LightRAG
        from lightrag.llm.ollama import ollama_model_complete
        from lightrag.utils import EmbeddingFunc

        _client = _ollama.AsyncClient(host=self.ollama_base_url)
        _model = self.embedding_model

        async def _embed(texts: list[str]) -> np.ndarray:
            r = await _client.embed(model=_model, input=texts)
            return np.array(r.embeddings)

        self._rag = LightRAG(
            working_dir=self.lightrag_working_dir,
            llm_model_func=ollama_model_complete,
            llm_model_name=self.llm_model,
            llm_model_kwargs={
                "host": self.ollama_base_url,
                "options": {"num_ctx": 32768},
            },
            embedding_func=EmbeddingFunc(
                embedding_dim=768,
                max_token_size=8192,
                func=_embed,
            ),
        )
        asyncio.run(self._rag.initialize_storages())
        logger.info("GraphRAGTool: LightRAG initialized from %s", self.lightrag_working_dir)

    def __call__(self, query: str, mode: str = "mix") -> list[dict]:
        if self._rag is None:
            self._init_rag()
        from lightrag import QueryParam
        try:
            result = self._rag.query(query, param=QueryParam(mode=mode))
            logger.info("GraphRAGTool: query complete (mode=%s)", mode)
            return [{"text": str(result), "source_path": "knowledge_graph", "score": 1.0}]
        except Exception as e:
            logger.warning("GraphRAGTool: query failed (%s)", e)
            return []


class KeywordRAGTool:
    """BM25 keyword search over the full ChromaDB corpus — lazy-indexed on first call."""

    def __init__(
        self,
        chroma_persist_path: str,
        chroma_collection_name: str,
        top_k: int,
    ):
        self.chroma_persist_path = chroma_persist_path
        self.chroma_collection_name = chroma_collection_name
        self.top_k = top_k
        self._index = None
        self._corpus: list[dict] = []

    def _build_index(self):
        import chromadb
        from rank_bm25 import BM25Okapi

        client = chromadb.PersistentClient(path=self.chroma_persist_path)
        collection = client.get_collection(name=self.chroma_collection_name)
        count = collection.count()

        # SQLite has a ~999 variable limit; paginate to avoid "too many SQL variables"
        PAGE = 5_000
        docs: list[str] = []
        metas: list[dict] = []
        for offset in range(0, count, PAGE):
            page = collection.get(
                limit=min(PAGE, count - offset),
                offset=offset,
                include=["documents", "metadatas"],
            )
            docs.extend(page["documents"])
            metas.extend(page["metadatas"])
        self._corpus = [
            {
                "text": text,
                "source_path": meta.get("source_path", "unknown"),
                "folder": meta.get("folder", "unknown"),
                "filename": meta.get("filename", "unknown"),
                "chunk_index": meta.get("chunk_index", 0),
            }
            for text, meta in zip(docs, metas)
        ]
        tokenized = [doc.lower().split() for doc in docs]
        self._index = BM25Okapi(tokenized)
        logger.info("KeywordRAGTool: BM25 index built over %d chunks", len(self._corpus))

    def __call__(self, query: str) -> list[dict]:
        if self._index is None:
            self._build_index()
        scores = self._index.get_scores(query.lower().split())
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[: self.top_k]
        return [
            {**self._corpus[i], "score": round(float(s), 4)}
            for i, s in indexed
            if s > 0
        ]
