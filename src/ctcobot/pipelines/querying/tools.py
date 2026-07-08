"""
RAG tool implementations for the querying pipeline.

Exposes callable tools that each return list[dict] with keys:
  {text, source_path, folder, filename, chunk_index, score}

Three tools:
  - VectorRAGTool   — HyDE + folder-priority turbovec retrieval + LLM rerank
  - KeywordRAGTool  — folder-scoped BM25 lexical retrieval
  - GraphRAGTool    — LightRAG knowledge-graph retrieval (entities/relations)
"""
import asyncio
import logging
import re

logger = logging.getLogger(__name__)


# ── BM25 keyword search ───────────────────────────────────────────────────────

# A small curated English stopword set. Kept inline to avoid an nltk dependency.
_ENGLISH_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when",
    "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "to", "from", "up", "down", "in", "out", "on",
    "off", "over", "under", "again", "further", "of", "as", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "do",
    "does", "did", "doing", "this", "that", "these", "those", "i", "you",
    "he", "she", "it", "we", "they", "what", "which", "who", "whom",
    "how", "why", "where", "their", "there", "here", "your", "our", "its",
    "can", "will", "would", "should", "could", "may", "might", "must",
    "not", "no", "yes", "so", "than", "too", "very", "just", "any", "all",
})

# Module-level cache: turbovec_persist_path -> (bm25_index, chunk_meta).
# Building BM25Okapi over ~48k tokenized chunks is expensive; the eval over
# 25 questions must build it once and reuse it.
_BM25_CACHE: dict[str, tuple] = {}

# Token = lowercase run of alphanumerics, optionally hyphen-joined.
# Preserves acronyms ("tntr"), numbers, and hyphenated terms ("9-box",
# "co-worker"). Lowercasing both corpus and query keeps acronym matching exact.
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def _bm25_tokenize(text: str) -> list[str]:
    """Tokenize text for BM25: lowercase, regex split, drop stopwords."""
    tokens = _TOKEN_RE.findall(text.lower())
    return [t for t in tokens if t not in _ENGLISH_STOPWORDS]


def _get_bm25_index(turbovec_persist_path: str):
    """
    Build (or fetch from cache) a BM25Okapi index over the handbook corpus.

    Corpus text and metadata come from the same ``meta.json`` sidecar the
    vector index uses, so BM25 and vector retrieval search identical chunks.

    Returns:
        (bm25_index, chunk_meta) where chunk_meta is the list of chunk dicts.
    """
    cached = _BM25_CACHE.get(turbovec_persist_path)
    if cached is not None:
        return cached

    from rank_bm25 import BM25Okapi
    from ctcobot.pipelines.querying.nodes import _load_turbovec

    logger.info("Building BM25 index from %s/meta.json ...", turbovec_persist_path)
    _index, chunk_meta = _load_turbovec(turbovec_persist_path)
    tokenized_corpus = [_bm25_tokenize(c["text"]) for c in chunk_meta]
    bm25 = BM25Okapi(tokenized_corpus)
    logger.info("BM25 index built over %d chunks.", len(chunk_meta))

    _BM25_CACHE[turbovec_persist_path] = (bm25, chunk_meta)
    return bm25, chunk_meta


class KeywordRAGTool:
    """Folder-scoped BM25 lexical retrieval.

    Complements the vector tool on exact-term / acronym / code questions
    (e.g. "TNTR", "9-box", "FMLA") where embeddings blur lexical matches.

    Restricting BM25 scoring to the question's top-ranked folders is the key
    safeguard against the v3.0 failure mode, where unscoped BM25 matched
    surface keywords across the 48k-chunk engineering-dominated corpus and
    polluted every HR answer.
    """

    def __init__(
        self,
        ollama_base_url: str,
        llm_model: str,
        turbovec_persist_path: str,
        top_k: int,
        top_folders: int,
        priority_folders: list[str] | None = None,
    ):
        self.ollama_base_url = ollama_base_url
        self.llm_model = llm_model
        self.turbovec_persist_path = turbovec_persist_path
        self.top_k = top_k
        self.top_folders = top_folders
        self.priority_folders = list(priority_folders or [])
        self.last_pre_rerank_sources: list[str] = []
        self.last_pre_rerank_chunks: list[dict] = []

    def __call__(self, query: str, ranked_folders: list[str] | None = None) -> list[dict]:
        import numpy as np
        from ctcobot.pipelines.querying.nodes import rank_folders

        bm25, chunk_meta = _get_bm25_index(self.turbovec_persist_path)

        if ranked_folders is None:
            ranked_folders = rank_folders(
                query, self.turbovec_persist_path, self.ollama_base_url, self.llm_model,
            )

        query_tokens = _bm25_tokenize(query)
        if not query_tokens:
            logger.warning("BM25 query had no tokens after filtering: %s", query[:60])
            self.last_pre_rerank_sources = []
            self.last_pre_rerank_chunks = []
            return []

        scores = bm25.get_scores(query_tokens)

        # Folder scoping: keep chunks whose folder is in the top-N ranked
        # folders, always unioned with the HR priority floor so canonical HR
        # folders survive folder-ranker variance. Empty set → skip the filter.
        priority_set = set(ranked_folders[:self.top_folders]) | set(self.priority_folders)
        order = np.argsort(scores)[::-1]

        candidates: list[int] = []
        if priority_set:
            candidates = [
                i for i in order
                if scores[i] > 0 and chunk_meta[i]["folder"] in priority_set
            ]
        if not candidates:
            if priority_set:
                logger.warning(
                    "BM25 folder filter (top=%s) yielded 0 hits; falling back to unscoped",
                    list(priority_set),
                )
            candidates = [i for i in order if scores[i] > 0]

        top_idx = candidates[:self.top_k]
        chunks = [
            {
                "text": chunk_meta[i]["text"],
                "source_path": chunk_meta[i]["source_path"],
                "folder": chunk_meta[i]["folder"],
                "filename": chunk_meta[i]["filename"],
                "chunk_index": chunk_meta[i]["chunk_index"],
                "score": round(float(scores[i]), 4),
            }
            for i in top_idx
        ]

        self.last_pre_rerank_sources = [c["source_path"] for c in chunks]
        self.last_pre_rerank_chunks = [
            {"source_path": c["source_path"], "chunk_index": c["chunk_index"]}
            for c in chunks
        ]
        logger.info(
            "BM25 retrieved %d chunks. Priority folders: %s. Top score: %s",
            len(chunks), list(priority_set),
            chunks[0]["score"] if chunks else "n/a",
        )
        return chunks


# ── Vector HyDE search ────────────────────────────────────────────────────────

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
        priority_folders: list[str] | None = None,
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
        self.priority_folders = list(priority_folders or [])
        self.last_pre_rerank_sources: list[str] = []
        self.last_pre_rerank_chunks: list[dict] = []
        self.last_ranked_folders: list[str] = []

    def __call__(self, query: str, ranked_folders: list[str] | None = None) -> list[dict]:
        from ctcobot.pipelines.querying.nodes import (
            generate_hyde_doc,
            embed_query,
            rank_folders,
            retrieve_chunks_folder_priority,
            rerank_chunks,
        )
        hyde = generate_hyde_doc(query, self.ollama_base_url, self.llm_model)
        emb = embed_query(hyde, self.ollama_base_url, self.embedding_model)
        if ranked_folders is None:
            ranked_folders = rank_folders(
                query, self.turbovec_persist_path, self.ollama_base_url, self.llm_model,
            )
        self.last_ranked_folders = ranked_folders
        raw = retrieve_chunks_folder_priority(
            emb,
            ranked_folders,
            self.turbovec_persist_path,
            self.top_k,
            self.top_folders,
            self.retrieve_oversample,
            self.priority_folders,
        )
        self.last_pre_rerank_sources = [c["source_path"] for c in raw]
        self.last_pre_rerank_chunks = [
            {"source_path": c["source_path"], "chunk_index": c["chunk_index"]}
            for c in raw
        ]
        return rerank_chunks(
            raw, query, self.reranker_model, self.rerank_top_n, self.ollama_base_url
        )


# ── LightRAG graph search ─────────────────────────────────────────────────────

# chunk_index namespace offset for graph-retrieved chunks. LightRAG re-chunks
# documents internally (~1200 tokens), so its chunk numbering is unrelated to
# the turbovec meta.json chunking; the offset keeps the agent's
# (source_path, chunk_index) dedup from ever colliding a graph chunk with a
# different-text vector/BM25 chunk of the same document.
_GRAPH_CHUNK_INDEX_BASE = 100_000

# Synthetic source for content that has no single origin document (the
# knowledge-graph facts summary, or chunks LightRAG returns without a path).
_GRAPH_KG_SOURCE = "lightrag://knowledge-graph"

# Module-level cache: working_dir -> (LightRAG instance, dedicated event loop).
# Initializing storages reloads every vector DB from disk; the eval loop must
# do that once, not per question. All coroutines for an instance run on its
# own loop so lightrag's loop handling stays consistent across calls.
_LIGHTRAG_CACHE: dict[str, tuple] = {}


def _graph_store_ready(working_dir: str) -> bool:
    """True if a *built* LightRAG graph store exists under ``working_dir``.

    A fresh or failed build leaves only ``kv_store_*.json`` stubs; a usable
    store has the entity/relationship artifacts written by a completed
    ``ctcobot index_graph`` run.
    """
    from pathlib import Path

    root = Path(working_dir)
    return any((root / name).exists() for name in (
        "vdb_entities.json",
        "vdb_relationships.json",
        "graph_chunk_entity_relation.graphml",
    ))


def _graph_data_to_chunks(
    result: dict,
    max_chunks: int,
    kg_summary_max_chars: int = 1200,
) -> list[dict]:
    """Map a LightRAG ``query_data`` payload onto the standard chunk schema.

    Two kinds of chunks come out:
      1. A single synthetic "knowledge-graph facts" chunk assembled from the
         retrieved entity and relationship descriptions (the graph's unique
         value: cross-document connections), capped at ``kg_summary_max_chars``.
      2. Up to ``max_chunks`` document chunks, each carrying the real
         ``file_path`` recorded at index time so answers can cite the source.

    Returns [] on a failure payload — the tool contributes nothing rather
    than sinking the question.
    """
    if not isinstance(result, dict) or result.get("status") != "success":
        return []
    data = result.get("data") or {}

    chunks_out: list[dict] = []

    # 1. Knowledge-graph facts summary (entities + relationships).
    lines: list[str] = []
    for e in data.get("entities") or []:
        name = (e.get("entity_name") or "").strip()
        desc = (e.get("description") or "").strip()
        if name and desc:
            lines.append(f"{name} ({e.get('entity_type', '?')}): {desc}")
    for r in data.get("relationships") or []:
        src, tgt = (r.get("src_id") or "").strip(), (r.get("tgt_id") or "").strip()
        desc = (r.get("description") or "").strip()
        if src and tgt and desc:
            lines.append(f"{src} -> {tgt}: {desc}")

    summary = ""
    for line in lines:
        if len(summary) + len(line) + 1 > kg_summary_max_chars:
            break
        summary += line + "\n"
    if summary.strip():
        chunks_out.append({
            "text": "Knowledge-graph facts extracted from the handbook:\n"
                    + summary.strip(),
            "source_path": _GRAPH_KG_SOURCE,
            "folder": "__graph__",
            "filename": "knowledge-graph",
            "chunk_index": _GRAPH_CHUNK_INDEX_BASE - 1,
            "score": 0.0,
        })

    # 2. Document chunks with real source paths.
    for i, ch in enumerate((data.get("chunks") or [])[:max_chunks]):
        text = (ch.get("content") or "").strip()
        if not text:
            continue
        fp = (ch.get("file_path") or "").strip()
        real = bool(fp) and fp.lower() not in {"unknown_source", "unknown"}
        chunks_out.append({
            "text": text,
            "source_path": fp if real else _GRAPH_KG_SOURCE,
            "folder": fp.split("/")[0] if real and "/" in fp else "__graph__",
            "filename": fp.rsplit("/", 1)[-1] if real else "knowledge-graph",
            "chunk_index": _GRAPH_CHUNK_INDEX_BASE + i,
            # Rank-derived placeholder: lightrag returns chunks already
            # ordered; raw graph scores are not comparable across tools anyway
            # and the merged rerank owns the final ordering.
            "score": round(1.0 / (i + 1), 4),
        })

    return chunks_out


class GraphRAGTool:
    """LightRAG knowledge-graph retrieval.

    Complements the vector and keyword tools on RELATIONSHIP questions —
    multi-hop questions spanning several policies, teams, or entities ("how
    does X affect Y", "who is responsible for X across Y") — where single-
    chunk retrieval misses the connection between documents.

    Retrieval-only integration: ``aquery_data`` returns the retrieved
    entities, relationships, and document chunks WITHOUT LightRAG's own
    answer generation, so the merged rerank + shared answer model stay in
    charge (one extra LLM call for lightrag's query-keyword extraction, none
    for generation).

    ``ranked_folders`` is accepted for interface parity but ignored — the
    graph is already scoped to the HR folders at index time
    (``lightrag.index_folders``).
    """

    def __init__(
        self,
        ollama_base_url: str,
        working_dir: str,
        llm_model: str,
        embedding_model: str,
        embedding_dim: int = 768,
        query_mode: str = "hybrid",
        top_k: int = 20,
        chunk_top_k: int = 5,
        num_ctx: int = 8192,
    ):
        self.ollama_base_url = ollama_base_url
        self.working_dir = working_dir
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        self.query_mode = query_mode
        self.top_k = top_k
        self.chunk_top_k = chunk_top_k
        self.num_ctx = num_ctx
        self.last_pre_rerank_sources: list[str] = []
        self.last_pre_rerank_chunks: list[dict] = []

    def _get_rag(self):
        """Build (or fetch from cache) the LightRAG instance + its event loop."""
        cached = _LIGHTRAG_CACHE.get(self.working_dir)
        if cached is not None:
            return cached

        import numpy as np
        import ollama as _ollama
        from lightrag import LightRAG
        from lightrag.kg.shared_storage import initialize_pipeline_status
        from lightrag.llm.ollama import ollama_model_complete
        from lightrag.utils import EmbeddingFunc

        logger.info("Loading LightRAG graph store from %s ...", self.working_dir)
        client = _ollama.AsyncClient(host=self.ollama_base_url)

        async def _embed(texts: list[str]) -> np.ndarray:
            r = await client.embed(model=self.embedding_model, input=texts)
            return np.array(r.embeddings)

        rag = LightRAG(
            working_dir=self.working_dir,
            llm_model_func=ollama_model_complete,
            llm_model_name=self.llm_model,
            llm_model_kwargs={
                "host": self.ollama_base_url,
                "options": {"num_ctx": self.num_ctx},
            },
            embedding_func=EmbeddingFunc(
                embedding_dim=self.embedding_dim,
                max_token_size=8192,
                func=_embed,
            ),
        )
        loop = asyncio.new_event_loop()
        loop.run_until_complete(rag.initialize_storages())
        loop.run_until_complete(initialize_pipeline_status())
        logger.info("LightRAG graph store loaded.")

        _LIGHTRAG_CACHE[self.working_dir] = (rag, loop)
        return rag, loop

    def __call__(self, query: str, ranked_folders: list[str] | None = None) -> list[dict]:
        from lightrag import QueryParam

        try:
            rag, loop = self._get_rag()
            param = QueryParam(
                mode=self.query_mode,
                top_k=self.top_k,
                chunk_top_k=self.chunk_top_k,
                # lightrag's own rerank binding is not configured; the agent's
                # merged rerank owns ordering.
                enable_rerank=False,
            )
            result = loop.run_until_complete(rag.aquery_data(query, param))
        except Exception as e:
            logger.warning("Graph retrieval failed (%s); contributing no chunks", e)
            self.last_pre_rerank_sources = []
            self.last_pre_rerank_chunks = []
            return []

        chunks = _graph_data_to_chunks(result, self.chunk_top_k)

        # Only doc-grounded chunks participate in retrieval metrics; the
        # synthetic knowledge-graph summary has no source document to credit.
        real = [c for c in chunks if c["source_path"] != _GRAPH_KG_SOURCE]
        self.last_pre_rerank_sources = [c["source_path"] for c in real]
        self.last_pre_rerank_chunks = [
            {"source_path": c["source_path"], "chunk_index": c["chunk_index"]}
            for c in real
        ]
        logger.info(
            "Graph retrieved %d chunks (%d doc-grounded). Mode: %s",
            len(chunks), len(real), self.query_mode,
        )
        return chunks


def maybe_graph_tool(lightrag_cfg: dict | None, ollama_base_url: str) -> GraphRAGTool | None:
    """Build a GraphRAGTool from the ``lightrag`` params block, or None.

    Returns None (with a log line saying why) when the block is missing,
    ``enabled`` is false, or the graph store has not been built yet — callers
    register the graph_rag tool only when it can actually serve queries.
    """
    cfg = lightrag_cfg or {}
    if not cfg.get("enabled", False):
        logger.info("LightRAG graph tool disabled in parameters.")
        return None
    working_dir = cfg.get("working_dir", "data/04_feature/lightrag_db")
    if not _graph_store_ready(working_dir):
        logger.warning(
            "LightRAG is enabled but no built graph store found at %s — "
            "run `ctcobot index_graph` first. Continuing without graph_rag.",
            working_dir,
        )
        return None
    return GraphRAGTool(
        ollama_base_url=ollama_base_url,
        working_dir=working_dir,
        llm_model=cfg.get("llm_model", "qwen3.5:4b"),
        embedding_model=cfg.get("embedding_model", "nomic-embed-text"),
        embedding_dim=int(cfg.get("embedding_dim", 768)),
        query_mode=cfg.get("query_mode", "hybrid"),
        top_k=int(cfg.get("top_k", 20)),
        chunk_top_k=int(cfg.get("chunk_top_k", 5)),
        num_ctx=int(cfg.get("num_ctx", 8192)),
    )
