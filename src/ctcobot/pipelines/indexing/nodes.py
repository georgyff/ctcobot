"""
Indexing pipeline nodes — ingest documents and build PageIndex hierarchical trees.
"""
import asyncio
import logging
import os
from pathlib import Path

import ollama as ollama_lib

from pageindex import PageIndexClient

logger = logging.getLogger(__name__)


def _patch_pageindex_for_ollama(ollama_base_url: str, pageindex_model: str) -> None:
    """
    Replace PageIndex's litellm-based LLM calls with direct Ollama SDK calls.

    litellm cannot connect to local Ollama (times out regardless of OLLAMA_API_BASE).
    The Ollama Python SDK works correctly. Both functions are patched at the module
    level so all callers within pageindex see the replacement immediately.

    generate_doc_description  → utils.llm_completion  (sync, one call per doc)
    generate_node_summary     → utils.llm_acompletion (async, one call per long section)
    """
    import pageindex.utils as _utils

    _client = ollama_lib.Client(host=ollama_base_url)

    def _sync_llm(model, prompt, chat_history=None, return_finish_reason=False):
        msgs = list(chat_history or []) + [{"role": "user", "content": prompt}]
        try:
            resp = _client.chat(model=pageindex_model, messages=msgs)
            content = resp["message"]["content"]
        except Exception as e:
            logger.warning(f"Ollama call failed during indexing: {e}")
            content = ""
        return (content, "finished") if return_finish_reason else content

    async def _async_llm(model, prompt):
        loop = asyncio.get_running_loop()
        msgs = [{"role": "user", "content": prompt}]
        try:
            resp = await loop.run_in_executor(
                None,
                lambda: _client.chat(model=pageindex_model, messages=msgs),
            )
            return resp["message"]["content"]
        except Exception as e:
            logger.warning(f"Ollama async call failed during indexing: {e}")
            return ""

    _utils.llm_completion = _sync_llm
    _utils.llm_acompletion = _async_llm


def ingest_documents(raw_data_path: str) -> list[dict]:
    """
    Walk the raw corpus directory and read all .md files.

    Args:
        raw_data_path: Path to the root of the markdown corpus.

    Returns:
        List of dicts with keys: path, raw_text, folder, filename.
    """
    root = Path(raw_data_path)

    if not root.exists():
        raise FileNotFoundError(f"Raw data path does not exist: {root}")

    md_files = sorted(root.rglob("*.md"))
    logger.info(f"Found {len(md_files)} .md files under {root}")

    documents = []
    skipped = 0

    for file_path in md_files:
        try:
            raw_text = file_path.read_text(encoding="utf-8", errors="ignore").strip()
            if not raw_text:
                skipped += 1
                continue

            relative_path = str(file_path.relative_to(root))
            parts = file_path.relative_to(root).parts
            folder = parts[0] if len(parts) > 1 else "__root__"

            documents.append({
                "path": relative_path,
                "raw_text": raw_text,
                "folder": folder,
                "filename": file_path.name,
            })

        except Exception as e:
            logger.warning(f"Could not read {file_path}: {e}")
            skipped += 1

    logger.info(f"Ingested {len(documents)} documents. Skipped {skipped}.")
    return documents


def pageindex_index_documents(
    raw_docs: list[dict],
    raw_data_path: str,
    pageindex_model: str,
    pageindex_workspace: str,
    ollama_base_url: str,
) -> dict:
    """
    Index all documents with PageIndex, building a hierarchical tree and generating
    LLM summaries for each node and a one-sentence document description.

    PageIndex's default LLM backend (litellm) cannot connect to local Ollama.
    This function patches litellm's call sites to use the Ollama Python SDK instead,
    then delegates to PageIndexClient.index() which runs:
      - md_to_tree() with if_add_node_summary='yes': section summaries via LLM
      - generate_doc_description(): one-sentence doc summary via LLM

    Both are cached to the pageindex_workspace/ JSON files. At query time,
    get_document_structure() returns the tree with summaries, giving the navigation
    LLM the context it needs to pick relevant sections.

    Supports incremental re-runs: files already in the workspace are skipped.

    Args:
        raw_docs:            Output of ingest_documents.
        raw_data_path:       Root path of the markdown corpus.
        pageindex_model:     Ollama model for summaries and descriptions (qwen3.5:4b).
        pageindex_workspace: Directory where PageIndex caches tree JSON files.
        ollama_base_url:     Ollama server URL.

    Returns:
        doc_registry: dict mapping doc_id -> {source_path, folder, filename, description}
    """
    _patch_pageindex_for_ollama(ollama_base_url, pageindex_model)

    client = PageIndexClient(
        model=pageindex_model,
        workspace=pageindex_workspace,
    )

    # Build reverse map of already-indexed absolute paths from the workspace
    indexed_paths: dict[str, str] = {
        doc["path"]: doc_id
        for doc_id, doc in client.documents.items()
        if doc.get("path")
    }

    root = Path(raw_data_path)
    doc_registry: dict = {}
    total = len(raw_docs)
    new_count = 0

    logger.info(
        f"PageIndex: indexing {total} documents with '{pageindex_model}' via Ollama..."
    )

    for i, doc_meta in enumerate(raw_docs):
        file_path = root / doc_meta["path"]
        abs_path = str(file_path.resolve())

        # ── Incremental: skip already-indexed files ───────────────────────────
        if abs_path in indexed_paths:
            doc_id = indexed_paths[abs_path]
            doc_registry[doc_id] = {
                "source_path": doc_meta["path"],
                "folder": doc_meta["folder"],
                "filename": doc_meta["filename"],
                "description": client.documents[doc_id].get("doc_description", ""),
            }
            continue

        # ── Index: build tree + generate summaries + description via Ollama ───
        try:
            doc_id = client.index(abs_path)
            description = client.documents[doc_id].get("doc_description", "")

            doc_registry[doc_id] = {
                "source_path": doc_meta["path"],
                "folder": doc_meta["folder"],
                "filename": doc_meta["filename"],
                "description": description,
            }
            indexed_paths[abs_path] = doc_id
            new_count += 1

            if (i + 1) % 50 == 0 or (i + 1) == total:
                logger.info(f"  Indexed {i + 1}/{total} ({new_count} new)")

        except Exception as e:
            logger.warning(f"Failed to index {doc_meta['path']}: {e}")

    logger.info(
        f"PageIndex indexing complete. "
        f"Registry: {len(doc_registry)}/{total} docs "
        f"({new_count} new, {len(doc_registry) - new_count} from cache)."
    )
    return doc_registry
