"""
LightRAG indexing pipeline node — builds the knowledge graph from cleaned documents.
"""
import asyncio
import logging

import numpy as np
import ollama as _ollama

logger = logging.getLogger(__name__)


def build_lightrag_index(
    cleaned_docs: list[dict],
    lightrag: dict,
    ollama_base_url: str,
) -> dict:
    """
    Build a LightRAG knowledge graph from full cleaned documents (not pre-chunked).

    Filters to the folders in ``lightrag.index_folders`` so only HR-relevant
    content is graph-indexed. LightRAG handles its own internal chunking at
    ~1200 tokens, which is appropriate for entity/relationship extraction
    (256-token pre-chunks are too small).

    Every document is inserted with its corpus-relative ``file_path``, so
    query-time results can cite the real source document — this is what lets
    the graph_rag tool participate in retrieval metrics.

    The whole build runs on ONE event loop (initialize → insert batches →
    finalize); the earlier version split this across loops and never called
    ``initialize_pipeline_status()``, which newer lightrag versions require
    before any insert.

    Args:
        cleaned_docs:    Full cleaned documents with 'text', 'folder', 'path' keys.
        lightrag:        The ``lightrag`` parameters block (see parameters.yml).
        ollama_base_url: Ollama server URL.

    Returns:
        Dict with status, num_docs indexed, and error count.
    """
    from lightrag import LightRAG
    from lightrag.kg.shared_storage import initialize_pipeline_status
    from lightrag.llm.ollama import ollama_model_complete
    from lightrag.utils import EmbeddingFunc
    from tqdm import tqdm

    working_dir = lightrag["working_dir"]
    folder_set = set(lightrag.get("index_folders") or [])
    docs = [
        d for d in cleaned_docs
        if (not folder_set or d.get("folder", "") in folder_set)
        and d.get("text", "").strip()
    ]
    logger.info(
        "Filtered to %d docs from folders %s (of %d total)",
        len(docs), sorted(folder_set) or "<all>", len(cleaned_docs),
    )

    batch_size = int(lightrag.get("insert_batch_size", 20))
    batches = [docs[i: i + batch_size] for i in range(0, len(docs), batch_size)]

    _client = _ollama.AsyncClient(host=ollama_base_url)

    async def _embed(texts: list[str]) -> np.ndarray:
        r = await _client.embed(model=lightrag["embedding_model"], input=texts)
        return np.array(r.embeddings)

    async def _build() -> int:
        rag = LightRAG(
            working_dir=working_dir,
            llm_model_func=ollama_model_complete,
            llm_model_name=lightrag["llm_model"],
            llm_model_kwargs={
                "host": ollama_base_url,
                "options": {"num_ctx": int(lightrag.get("num_ctx", 8192))},
            },
            embedding_func=EmbeddingFunc(
                embedding_dim=int(lightrag.get("embedding_dim", 768)),
                max_token_size=8192,
                func=_embed,
            ),
            max_parallel_insert=int(lightrag.get("max_parallel_insert", 2)),
        )
        await rag.initialize_storages()
        await initialize_pipeline_status()

        errors = 0
        with tqdm(batches, desc="Indexing docs", unit="batch", dynamic_ncols=True) as pbar:
            for i, batch in enumerate(pbar):
                try:
                    await rag.ainsert(
                        [d["text"] for d in batch],
                        file_paths=[d["path"] for d in batch],
                    )
                except Exception as e:
                    errors += 1
                    logger.warning("Batch %d failed: %s", i, e)
                docs_done = min((i + 1) * batch_size, len(docs))
                status = "ok" if errors == 0 else f"{errors} err"
                pbar.set_postfix(docs=f"{docs_done}/{len(docs)}", status=status)

        await rag.finalize_storages()
        return errors

    errors = asyncio.run(_build())

    logger.info("LightRAG indexing complete (%d errors).", errors)
    return {
        "status": "indexed",
        "num_docs": len(docs),
        "errors": errors,
        "working_dir": working_dir,
    }
