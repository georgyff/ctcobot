"""
LightRAG indexing pipeline node — builds the knowledge graph from chunked documents.
"""
import asyncio
import logging

import numpy as np
import ollama as _ollama

logger = logging.getLogger(__name__)


def build_lightrag_index(
    cleaned_docs: list[dict],
    lightrag_index_folders: list[str],
    lightrag_working_dir: str,
    ollama_base_url: str,
    llm_model: str,
    embedding_model: str,
) -> dict:
    """
    Build a LightRAG knowledge graph from full cleaned documents (not pre-chunked).

    Filters to the specified folders so only HR-relevant content is graph-indexed.
    LightRAG handles its own internal chunking at ~1200 tokens, which is appropriate
    for entity/relationship extraction (256-token pre-chunks are too small).

    Args:
        cleaned_docs:           Full cleaned documents with 'text', 'folder' keys.
        lightrag_index_folders: Handbook folders to include (e.g. people-policies).
        lightrag_working_dir:   Path for LightRAG graph storage.
        ollama_base_url:        Ollama server URL.
        llm_model:              LLM model used for entity/relationship extraction.
        embedding_model:        Embedding model for vector components of the graph.

    Returns:
        Dict with status and num_docs indexed.
    """
    from lightrag import LightRAG
    from lightrag.llm.ollama import ollama_model_complete
    from lightrag.utils import EmbeddingFunc

    _client = _ollama.AsyncClient(host=ollama_base_url)

    async def _embed(texts: list[str]) -> np.ndarray:
        r = await _client.embed(model=embedding_model, input=texts)
        return np.array(r.embeddings)

    rag = LightRAG(
        working_dir=lightrag_working_dir,
        llm_model_func=ollama_model_complete,
        llm_model_name=llm_model,
        llm_model_kwargs={
            "host": ollama_base_url,
            "options": {"num_ctx": 8192},
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=768,
            max_token_size=8192,
            func=_embed,
        ),
        max_parallel_insert=2,
    )
    asyncio.run(rag.initialize_storages())

    from tqdm import tqdm

    folder_set = set(lightrag_index_folders)
    docs = [
        d for d in cleaned_docs
        if d.get("folder", "") in folder_set and d.get("text", "").strip()
    ]
    logger.info(
        "Filtered to %d docs from folders %s (of %d total)",
        len(docs), sorted(folder_set), len(cleaned_docs),
    )
    texts = [d["text"] for d in docs]

    batch_size = 20
    batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
    errors = 0
    with tqdm(batches, desc="Indexing docs", unit="batch", dynamic_ncols=True) as pbar:
        for i, batch in enumerate(pbar):
            try:
                rag.insert(batch)
            except Exception as e:
                errors += 1
                logger.warning("Batch %d failed: %s", i, e)
            docs_done = min((i + 1) * batch_size, len(texts))
            status = "ok" if errors == 0 else f"{errors} err"
            pbar.set_postfix(docs=f"{docs_done}/{len(texts)}", status=status)

    logger.info("LightRAG indexing complete.")
    return {"status": "indexed", "num_docs": len(texts), "working_dir": lightrag_working_dir}
