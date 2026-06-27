"""
RAG tool implementations for the querying pipeline.

Exposes callable tools that each return list[dict] with keys:
  {text, source_path, folder, filename, chunk_index, score}

Two tools:
  - VectorRAGTool   — HyDE + folder-priority turbovec retrieval + LLM rerank
  - KeywordRAGTool  — folder-scoped BM25 lexical retrieval
"""
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
        deprioritize_path_patterns: list[str] | None = None,
        entity_penalty_factor: float = 1.0,
    ):
        self.ollama_base_url = ollama_base_url
        self.llm_model = llm_model
        self.turbovec_persist_path = turbovec_persist_path
        self.top_k = top_k
        self.top_folders = top_folders
        self.priority_folders = list(priority_folders or [])
        # GEN.6: same entity-page de-prioritization the vector path uses, so
        # location/entity leaf pages don't leak into the merged set via BM25.
        self.deprioritize_path_patterns = list(deprioritize_path_patterns or [])
        self.entity_penalty_factor = entity_penalty_factor
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

        # GEN.6: entity-page de-prioritization (mirrors the vector path). Demote
        # location/entity leaf pages so they stop crowding general questions; only
        # touch chunks BM25 actually scored, and never an `_index.md` overview.
        patterns = self.deprioritize_path_patterns
        if patterns and self.entity_penalty_factor < 1.0:
            for i in np.nonzero(scores)[0]:
                sp = chunk_meta[i]["source_path"]
                if not sp.endswith("_index.md") and any(p in sp for p in patterns):
                    scores[i] *= self.entity_penalty_factor

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
        deprioritize_path_patterns: list[str] | None = None,
        entity_penalty_factor: float = 1.0,
        vector_rerank_top_n: int | None = None,
    ):
        self.ollama_base_url = ollama_base_url
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.turbovec_persist_path = turbovec_persist_path
        self.top_k = top_k
        self.reranker_model = reranker_model
        self.rerank_top_n = rerank_top_n
        # GEN.5: how many chunks the tool keeps after its own (weak) rerank before
        # handing off to the agent's 4b reranker. Default = top_k (pass-through:
        # drop nothing, let the 4b reranker own the final selection).
        self.vector_rerank_top_n = vector_rerank_top_n or top_k
        self.top_folders = top_folders
        self.retrieve_oversample = retrieve_oversample
        self.priority_folders = list(priority_folders or [])
        self.deprioritize_path_patterns = list(deprioritize_path_patterns or [])
        self.entity_penalty_factor = entity_penalty_factor
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
            self.deprioritize_path_patterns,
            self.entity_penalty_factor,
        )
        self.last_pre_rerank_sources = [c["source_path"] for c in raw]
        self.last_pre_rerank_chunks = [
            {"source_path": c["source_path"], "chunk_index": c["chunk_index"]}
            for c in raw
        ]
        return rerank_chunks(
            raw, query, self.reranker_model, self.vector_rerank_top_n,
            self.ollama_base_url,
        )
