"""
Querying pipeline nodes — embed, retrieve, prompt, generate.
"""
import logging

logger = logging.getLogger(__name__)


def embed_query(
    question: str,
    ollama_base_url: str,
    embedding_model: str,
) -> list[float]:
    """
    Embed a user query using Ollama.

    Args:
        question:        The user's question string.
        ollama_base_url: Ollama server URL.
        embedding_model: Ollama embedding model name.

    Returns:
        Embedding vector as a list of floats.
    """
    import ollama

    client = ollama.Client(host=ollama_base_url)
    response = client.embeddings(model=embedding_model, prompt=question)
    embedding = response["embedding"]
    logger.info(f"Query embedded. Dims: {len(embedding)}")
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

    logger.info(f"Retrieved {len(chunks)} chunks. Top score: {chunks[0]['score']}")
    return chunks


def build_prompt(
    question: str,
    chunks: list[dict],
) -> dict:
    """
    Assemble the RAG prompt from the question and retrieved chunks.

    Args:
        question: The user's question.
        chunks:   Retrieved chunks from retrieve_chunks.

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
                "score": chunk["score"],
            })

    result = {
        "question": prompt_data["question"],
        "answer": answer,
        "sources": sources,
        "model": llm_model,
    }

    logger.info(f"Answer generated. Sources: {[s['source_path'] for s in sources]}")
    return result