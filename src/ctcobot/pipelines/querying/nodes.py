"""
Querying pipeline nodes — hyde, embed, retrieve, rerank, prompt, generate.
"""
import json
import logging
from pathlib import Path

import numpy as np
from turbovec import TurboQuantIndex

logger = logging.getLogger(__name__)

REWRITE_SYSTEM_PROMPT = (
    "You are a query expansion assistant for an HR policy retrieval system. "
    "Rewrite the question below to improve document retrieval. "
    "IMPORTANT: Always preserve the main policy topic as the primary subject. "
    "You may add synonyms for secondary terms and expand acronyms, "
    "but never let synonym expansion shift the core topic. "
    "Output only the rewritten query, nothing else."
)


def generate_hyde_doc(
    question: str,
    ollama_base_url: str,
    llm_model: str,
) -> str:
    """
    Generate a hypothetical document for HyDE retrieval.

    Instead of embedding the query, embeds a plausible policy paragraph that
    would answer the question. The paragraph is in the same semantic space as
    actual handbook chunks, improving cosine-similarity retrieval.

    Args:
        question:        The original user question.
        ollama_base_url: Ollama server URL.
        llm_model:       Model used to generate the hypothetical document.

    Returns:
        A short policy paragraph (3-5 sentences) as a plain string.
    """
    import ollama
    from ctcobot.prompt_templates import HYDE_SYSTEM_PROMPT

    client = ollama.Client(host=ollama_base_url)
    response = client.chat(
        model=llm_model,
        messages=[
            {"role": "system", "content": HYDE_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        think=False,
    )
    hyde_doc = response["message"]["content"].strip()
    logger.info("HyDE doc generated for: %s", question[:60])
    return hyde_doc


def rewrite_query(
    question: str,
    ollama_base_url: str,
    llm_model: str,
) -> str:
    """
    Rewrite and expand the user question for better embedding retrieval.

    Args:
        question:        The original user question.
        ollama_base_url: Ollama server URL.
        llm_model:       Ollama model used for rewriting.

    Returns:
        Expanded query string.
    """
    import ollama

    client = ollama.Client(host=ollama_base_url)
    response = client.chat(
        model=llm_model,
        messages=[
            {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        think=False,
    )
    expanded = response["message"]["content"].strip()
    logger.info("Query rewritten: %s → %s", question[:60], expanded[:80])
    return expanded


SUB_QUERY_SYSTEM_PROMPT = (
    "You are a search query decomposition assistant for an HR policy retrieval system. "
    "Given a user question, generate exactly {n} specific sub-questions that together cover "
    "different aspects or phrasings of the original question. "
    "Each sub-question should be phrased as a standalone question a user might ask. "
    "Output ONLY a JSON array of strings, nothing else.\n"
    'Example: ["How does X work?", "What are the requirements for X?"]'
)


def generate_sub_queries(
    question: str,
    ollama_base_url: str,
    llm_model: str,
    num_sub_queries: int,
) -> list[str]:
    """
    Decompose a question into focused sub-questions for multi-query retrieval.

    Each sub-question covers a different aspect or phrasing, so that when
    embedded independently they pull different chunks from the same source
    document — improving recall without changing the index.

    Args:
        question:        The original user question.
        ollama_base_url: Ollama server URL.
        llm_model:       Model used to generate sub-questions.
        num_sub_queries: How many sub-questions to generate.

    Returns:
        List of sub-question strings (may be shorter than requested on parse failure).
    """
    import json
    import re
    import ollama

    client = ollama.Client(host=ollama_base_url)
    response = client.chat(
        model=llm_model,
        messages=[
            {"role": "system", "content": SUB_QUERY_SYSTEM_PROMPT.format(n=num_sub_queries)},
            {"role": "user", "content": question},
        ],
        think=False,
    )
    raw = response["message"]["content"].strip()
    match = re.search(r'\[.*?\]', raw, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, list):
                sub_qs = [str(q) for q in parsed[:num_sub_queries]]
                logger.info("Generated %d sub-queries for: %s", len(sub_qs), question[:60])
                return sub_qs
        except json.JSONDecodeError:
            pass
    logger.warning("Sub-query parse failed, falling back to no sub-queries")
    return []


def embed_query(
    question: str,
    ollama_base_url: str,
    embedding_model: str,
) -> list[float]:
    """
    Embed a query string using Ollama.

    Args:
        question:        The query string to embed (may be rewritten).
        ollama_base_url: Ollama server URL.
        embedding_model: Ollama embedding model name.

    Returns:
        Embedding vector as a list of floats.
    """
    import ollama

    client = ollama.Client(host=ollama_base_url)
    response = client.embeddings(model=embedding_model, prompt=question)
    embedding = response["embedding"]
    logger.info("Query embedded. Dims: %d", len(embedding))
    return embedding


def _load_turbovec(turbovec_persist_path: str):
    """Load turbovec index and metadata sidecar from disk."""
    meta_path = Path(turbovec_persist_path) / "meta.json"
    index_path = str(Path(turbovec_persist_path) / "index.tq")
    with open(meta_path) as f:
        meta_doc = json.load(f)
    index = TurboQuantIndex.load(index_path)
    return index, meta_doc["chunks"]


def _list_folders(turbovec_persist_path: str) -> list[str]:
    """Return the sorted set of top-level folder names present in the index."""
    meta_path = Path(turbovec_persist_path) / "meta.json"
    with open(meta_path) as f:
        meta_doc = json.load(f)
    folders = {c.get("folder", "") for c in meta_doc["chunks"]}
    folders.discard("")
    return sorted(folders)


FOLDER_RANK_TIMEOUT_SECONDS = 90


def rank_folders(
    question: str,
    turbovec_persist_path: str,
    ollama_base_url: str,
    llm_model: str,
) -> list[str]:
    """
    Use the LLM to rank handbook folders by likelihood of containing the answer.

    Uses Ollama's ``format="json"`` to constrain output to a valid JSON object
    of shape ``{"ranked": ["folder1", ...]}``. On any failure (parse error,
    timeout, hang, empty list) returns an **empty list** as a signal that the
    folder filter should be skipped — ``retrieve_chunks_folder_priority`` then
    falls back to plain top-k retrieval over the full index. This avoids the
    earlier bug where a parse failure silently restricted retrieval to the
    alphabetical first five folders (``about``, ``acquisitions``, ...).

    Args:
        question:              The original user question.
        turbovec_persist_path: Directory containing meta.json.
        ollama_base_url:       Ollama server URL.
        llm_model:             Model used for folder ranking.

    Returns:
        Ranked folder list (MOST relevant first), or ``[]`` to signal
        "skip the folder filter for this query".
    """
    import ollama
    from ctcobot.prompt_templates import FOLDER_RANK_SYSTEM_PROMPT

    folders = _list_folders(turbovec_persist_path)
    user_msg = (
        f"Question: {question}\n\n"
        f"Folders:\n" + "\n".join(f"- {f}" for f in folders)
    )

    client = ollama.Client(host=ollama_base_url, timeout=FOLDER_RANK_TIMEOUT_SECONDS)
    try:
        response = client.chat(
            model=llm_model,
            messages=[
                {"role": "system", "content": FOLDER_RANK_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            format="json",
            think=False,
        )
        raw = response["message"]["content"].strip()
        parsed = json.loads(raw)
    except Exception as e:
        logger.warning(
            "Folder ranking failed (%s); skipping folder filter for this query",
            e,
        )
        return []

    # Accept {"ranked": [...]}, bare [...], or {"<anything>": [...]}.
    if isinstance(parsed, list):
        ranked = parsed
    elif isinstance(parsed, dict):
        ranked = parsed.get("ranked")
        if not isinstance(ranked, list):
            ranked = next(
                (v for v in parsed.values() if isinstance(v, list)),
                [],
            )
    else:
        ranked = []

    folder_set = set(folders)
    seen: set[str] = set()
    ordered: list[str] = []
    for f in ranked:
        if isinstance(f, str) and f in folder_set and f not in seen:
            seen.add(f)
            ordered.append(f)

    if not ordered:
        logger.warning(
            "LLM returned no recognized folder names; skipping folder filter"
        )
        return []

    # Pad with any folders the LLM omitted so downstream priority slicing is
    # stable regardless of how many names the LLM emitted.
    for f in folders:
        if f not in seen:
            ordered.append(f)

    logger.info("Folder ranking (top 5): %s", ordered[:5])
    return ordered


def retrieve_chunks_folder_priority(
    query_embedding: list[float],
    ranked_folders: list[str],
    turbovec_persist_path: str,
    top_k: int,
    top_folders: int,
    retrieve_oversample: int,
    priority_folders: list[str] | None = None,
) -> list[dict]:
    """
    Retrieve chunks restricted to the top-N ranked folders.

    Oversamples (top_k × retrieve_oversample) from the full index, filters to
    chunks whose folder is in the top-N ranked folders, and returns the top
    ``top_k`` survivors in score order. Falls back to the unfiltered top_k if
    the folder filter would yield zero chunks.

    ``priority_folders`` are always unioned into the filter set, so the
    canonical HR folders remain candidates even when the LLM folder ranker
    omits them (the v5.0 talent-assessment misses) or fails entirely.

    Args:
        query_embedding:       Embedded query vector (will be L2-normalized).
        ranked_folders:        Folders ordered MOST→LEAST relevant (from
                               rank_folders).
        turbovec_persist_path: Directory containing index.tq and meta.json.
        top_k:                 Final number of chunks to return.
        top_folders:           Number of top-ranked folders to keep.
        retrieve_oversample:   Multiplier on top_k for the index pre-search.
        priority_folders:      Folders always included in the filter set.

    Returns:
        List of chunk dicts with keys: text, source_path, folder,
        filename, chunk_index, score.
    """
    index, chunk_meta = _load_turbovec(turbovec_persist_path)

    floor = set(priority_folders or [])
    # The filter set is the LLM's top-N ranked folders plus the always-on HR
    # floor. Only skip filtering when BOTH are empty (ranking failed and no
    # floor configured).
    priority_set = set(ranked_folders[:top_folders]) | floor
    skip_filter = not priority_set

    q = np.array(query_embedding, dtype=np.float32)
    norm = np.linalg.norm(q)
    if norm > 0:
        q = q / norm

    if skip_filter:
        oversample_k = min(top_k, len(chunk_meta))
    else:
        oversample_k = max(top_k, top_k * retrieve_oversample)
        oversample_k = min(oversample_k, len(chunk_meta))
    scores, indices = index.search(q[np.newaxis, :], k=oversample_k)

    all_hits = [
        {
            "text": chunk_meta[i]["text"],
            "source_path": chunk_meta[i]["source_path"],
            "folder": chunk_meta[i]["folder"],
            "filename": chunk_meta[i]["filename"],
            "chunk_index": chunk_meta[i]["chunk_index"],
            "score": round(float(s), 4),
        }
        for s, i in zip(scores[0], indices[0])
    ]

    if skip_filter:
        chunks = all_hits[:top_k]
        logger.info(
            "Folder filter SKIPPED (ranking unavailable); retrieved %d chunks. "
            "Top score: %s",
            len(chunks), chunks[0]["score"] if chunks else "n/a",
        )
        return chunks

    filtered = [c for c in all_hits if c["folder"] in priority_set]
    if not filtered:
        logger.warning(
            "Folder filter (top=%s) yielded 0 chunks; falling back to unfiltered top_k",
            list(priority_set),
        )
        chunks = all_hits[:top_k]
    else:
        chunks = filtered[:top_k]

    logger.info(
        "Folder-priority retrieved %d chunks. Priority folders: %s. Top score: %s",
        len(chunks), list(priority_set),
        chunks[0]["score"] if chunks else "n/a",
    )
    return chunks


def retrieve_chunks(
    query_embedding: list[float],
    turbovec_persist_path: str,
    top_k: int,
) -> list[dict]:
    """
    Retrieve top-k most relevant chunks from the turbovec index.

    Args:
        query_embedding:       Embedded query vector (will be L2-normalized).
        turbovec_persist_path: Directory containing index.tq and meta.json.
        top_k:                 Number of chunks to retrieve.

    Returns:
        List of chunk dicts with keys: text, source_path, folder,
        filename, chunk_index, score.
    """
    index, chunk_meta = _load_turbovec(turbovec_persist_path)

    q = np.array(query_embedding, dtype=np.float32)
    norm = np.linalg.norm(q)
    if norm > 0:
        q = q / norm

    scores, indices = index.search(q[np.newaxis, :], k=top_k)

    chunks = [
        {
            "text": chunk_meta[i]["text"],
            "source_path": chunk_meta[i]["source_path"],
            "folder": chunk_meta[i]["folder"],
            "filename": chunk_meta[i]["filename"],
            "chunk_index": chunk_meta[i]["chunk_index"],
            "score": round(float(s), 4),
        }
        for s, i in zip(scores[0], indices[0])
    ]

    logger.info("Retrieved %d chunks. Top score: %s", len(chunks), chunks[0]["score"] if chunks else "n/a")
    return chunks


def retrieve_chunks_multi(
    query_embedding: list[float],
    sub_queries: list[str],
    ollama_base_url: str,
    embedding_model: str,
    turbovec_persist_path: str,
    top_k: int,
) -> list[dict]:
    """
    Retrieve chunks using the HyDE embedding plus one embedding per sub-query.

    Each query searches the turbovec index independently for top_k results.
    Results are merged and deduplicated by (source_path, chunk_index), keeping
    the highest score per chunk.

    Args:
        query_embedding:       Embedded HyDE document vector.
        sub_queries:           Sub-questions to also retrieve for (may be empty).
        ollama_base_url:       Ollama server URL.
        embedding_model:       Ollama embedding model name.
        turbovec_persist_path: Directory containing index.tq and meta.json.
        top_k:                 Number of chunks to retrieve per query.

    Returns:
        Deduplicated list of chunk dicts sorted by score descending.
    """
    import ollama as _ollama

    index, chunk_meta = _load_turbovec(turbovec_persist_path)
    oc = _ollama.Client(host=ollama_base_url)

    raw_embeddings = [query_embedding]
    for sub_q in sub_queries:
        resp = oc.embeddings(model=embedding_model, prompt=sub_q)
        raw_embeddings.append(resp["embedding"])

    seen: dict[tuple, dict] = {}
    for raw_emb in raw_embeddings:
        q = np.array(raw_emb, dtype=np.float32)
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm
        scores, indices = index.search(q[np.newaxis, :], k=top_k)
        for s, i in zip(scores[0], indices[0]):
            m = chunk_meta[i]
            key = (m["source_path"], m["chunk_index"])
            score = round(float(s), 4)
            if key not in seen or score > seen[key]["score"]:
                seen[key] = {
                    "text": m["text"],
                    "source_path": m["source_path"],
                    "folder": m["folder"],
                    "filename": m["filename"],
                    "chunk_index": m["chunk_index"],
                    "score": score,
                }

    chunks = sorted(seen.values(), key=lambda c: c["score"], reverse=True)
    logger.info(
        "Multi-query retrieved %d unique chunks from %d queries.",
        len(chunks), len(raw_embeddings),
    )
    return chunks


RERANK_PROMPT = (
    "You are a relevance reranker. Given a question and numbered excerpts, "
    "rank them from MOST to LEAST relevant to answering the specific question. "
    "Prefer excerpts that DIRECTLY answer the question — containing the "
    "actual fact, definition, procedure, or contact information being asked "
    "for. Demote excerpts that only mention the topic in passing or that "
    "describe a related but different policy. "
    "Do not bias toward chunks with more numbers or longer text; "
    "judge each chunk by whether its content answers THIS question. "
    "Include every excerpt number exactly once. "
    "Output ONLY the JSON array, nothing else.\n\n"
    "Example output: [3, 1, 5, 2, 4]"
)

# Bound the reranker call so a stalled/runaway generation can't hang the
# pipeline; on timeout rerank_chunks falls back to the original score order.
RERANK_TIMEOUT_SECONDS = 180

# Generous bound for answer generation (real answers can be long).
GENERATE_TIMEOUT_SECONDS = 300


def rerank_chunks(
    chunks: list[dict],
    question: str,
    reranker_model: str,
    rerank_top_n: int,
    ollama_base_url: str,
) -> list[dict]:
    """
    Rerank retrieved chunks using a local Ollama model (listwise).

    Args:
        chunks:          Chunks from retrieve_chunks.
        question:        The original user question (not rewritten).
        reranker_model:  Ollama model used for reranking.
        rerank_top_n:    Number of top chunks to keep after reranking.
        ollama_base_url: Ollama server URL.

    Returns:
        Top-N chunks in relevance order.
    """
    import json
    import re
    import ollama

    if not chunks:
        return chunks

    excerpt_lines = "\n\n".join(
        f"[{i+1}] {chunk['text'].strip()[:700]}"
        for i, chunk in enumerate(chunks)
    )
    user_msg = f"Question: {question}\n\nExcerpts:\n{excerpt_lines}"

    # The reranker only needs to emit a short JSON array of indices. Cap the
    # output (num_predict) so a heavier model can't run away into a multi-minute
    # generation, and bound the call with a timeout. On any failure (timeout,
    # parse error) we fall back to the original score order below.
    client = ollama.Client(host=ollama_base_url, timeout=RERANK_TIMEOUT_SECONDS)
    try:
        response = client.chat(
            model=reranker_model,
            messages=[
                {"role": "system", "content": RERANK_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            think=False,
            options={"num_predict": 256, "temperature": 0.0},
        )
        raw = response["message"]["content"].strip()
        match = re.search(r'\[[\d,\s]+\]', raw)
        ranked_indices = json.loads(match.group()) if match else list(range(1, len(chunks) + 1))
    except Exception as e:
        logger.warning("Reranker failed (%s), using original order", e)
        ranked_indices = list(range(1, len(chunks) + 1))

    seen: set[int] = set()
    ordered: list[dict] = []
    for idx in ranked_indices:
        i = idx - 1
        if 0 <= i < len(chunks) and i not in seen:
            seen.add(i)
            ordered.append(chunks[i])

    # Append any chunks the model omitted (shouldn't happen, but safe)
    for i, chunk in enumerate(chunks):
        if i not in seen:
            ordered.append(chunk)

    top = ordered[:rerank_top_n]
    logger.info("Reranked %d → %d chunks.", len(chunks), len(top))
    return top


def build_prompt(
    question: str,
    chunks: list[dict],
) -> dict:
    """
    Assemble the RAG prompt from the question and retrieved chunks.

    Args:
        question: The original user question.
        chunks:   Retrieved (and optionally reranked) chunks.

    Returns:
        Dict with keys: system_prompt, user_prompt, question, chunks.
    """
    from ctcobot.prompt_templates import (
        SYSTEM_PROMPT,
        format_context,
        format_rag_prompt,
    )

    context = format_context(chunks)
    user_prompt = format_rag_prompt(question, context)

    return {
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": user_prompt,
        "question": question,
        "chunks": chunks,
    }


def generate_answer(
    prompt_data: dict,
    ollama_base_url: str,
    llm_model: str,
) -> dict:
    """
    Generate an answer using the Ollama LLM.

    Args:
        prompt_data:     Output of build_prompt.
        ollama_base_url: Ollama server URL.
        llm_model:       Ollama LLM model name.

    Returns:
        Dict with keys: question, answer, sources, model.
        
    Do not rely on your training knowledge; only state what appears verbatim in the excerpts.
    """
    import ollama

    # Generous bound so answer generation can't hang the run indefinitely.
    client = ollama.Client(host=ollama_base_url, timeout=GENERATE_TIMEOUT_SECONDS)

    response = client.chat(
        model=llm_model,
        messages=[
            {"role": "system", "content": prompt_data["system_prompt"]},
            {"role": "user", "content": prompt_data["user_prompt"]},
        ],
        think=False,
        # Low temperature for coherent, reproducible answers (avoids the
        # high-temperature self-contradiction loop); num_predict bounds any
        # runaway repetition while still allowing full answers.
        options={"temperature": 0.2, "num_predict": 800},
    )

    answer = response["message"]["content"].strip()

    # Deduplicate sources
    seen = set()
    sources = []
    for chunk in prompt_data["chunks"]:
        src = chunk["source_path"]
        if src not in seen:
            seen.add(src)
            sources.append({
                "source_path": src,
                "score": chunk.get("rerank_score", chunk["score"]),
            })

    result = {
        "question": prompt_data["question"],
        "answer": answer,
        "sources": sources,
        "model": llm_model,
    }

    logger.info("Answer generated. Sources: %s", [s["source_path"] for s in sources])
    return result
