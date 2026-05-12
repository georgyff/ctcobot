"""
Querying pipeline nodes — rewrite, embed, retrieve, rerank, prompt, generate.
"""
import logging

logger = logging.getLogger(__name__)

REWRITE_SYSTEM_PROMPT = (
    "You are a query expansion assistant for an HR policy retrieval system. "
    "Rewrite the question below to improve document retrieval. "
    "IMPORTANT: Always preserve the main policy topic as the primary subject. "
    "You may add synonyms for secondary terms and expand acronyms, "
    "but never let synonym expansion shift the core topic. "
    "Output only the rewritten query, nothing else."
)


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


def retrieve_chunks(
    query_embedding: list[float],
    chroma_persist_path: str,
    chroma_collection_name: str,
    top_k: int,
) -> list[dict]:
    """
    Retrieve top-k most relevant chunks from ChromaDB.

    Args:
        query_embedding:        Embedded query vector.
        chroma_persist_path:    Path to ChromaDB persistent store.
        chroma_collection_name: ChromaDB collection name.
        top_k:                  Number of chunks to retrieve.

    Returns:
        List of chunk dicts with keys: text, source_path, folder,
        filename, chunk_index, score.
    """
    import chromadb

    client = chromadb.PersistentClient(path=chroma_persist_path)
    collection = client.get_collection(name=chroma_collection_name)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for text, metadata, distance in zip(documents, metadatas, distances):
        chunks.append({
            "text": text,
            "source_path": metadata.get("source_path", "unknown"),
            "folder": metadata.get("folder", "unknown"),
            "filename": metadata.get("filename", "unknown"),
            "chunk_index": metadata.get("chunk_index", 0),
            "score": round(1 - distance, 4),  # cosine similarity
        })

    logger.info("Retrieved %d chunks. Top score: %s", len(chunks), chunks[0]["score"] if chunks else "n/a")
    return chunks


RERANK_PROMPT = (
    "You are a relevance reranker. Given a question and numbered excerpts, "
    "output a JSON array of excerpt numbers ranked from MOST to LEAST relevant. "
    "Include every number. Output ONLY the JSON array, nothing else.\n\n"
    "Example output: [3, 1, 5, 2, 4]"
)


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
        f"[{i+1}] {chunk['text'].strip()[:400]}"
        for i, chunk in enumerate(chunks)
    )
    user_msg = f"Question: {question}\n\nExcerpts:\n{excerpt_lines}"

    client = ollama.Client(host=ollama_base_url)
    try:
        response = client.chat(
            model=reranker_model,
            messages=[
                {"role": "system", "content": RERANK_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            think=False,
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
    """
    import ollama

    client = ollama.Client(host=ollama_base_url)

    response = client.chat(
        model=llm_model,
        messages=[
            {"role": "system", "content": prompt_data["system_prompt"]},
            {"role": "user", "content": prompt_data["user_prompt"]},
        ],
        think=False,
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
