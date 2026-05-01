"""
Querying pipeline nodes — PageIndex retrieval, prompt assembly, answer generation.
"""
import dataclasses
import json
import logging
import os
import re

import ollama as ollama_lib
from pageindex import PageIndexClient  # noqa: E402

from ctcobot.prompt_templates import SYSTEM_PROMPT, format_context, format_rag_prompt

logger = logging.getLogger(__name__)

_MAX_DOCS_IN_PROMPT = 200


@dataclasses.dataclass
class _RetrievalCtx:
    pageindex_client: PageIndexClient
    ollama_client: ollama_lib.Client
    model: str
    top_sections: int


def _parse_numbers(raw: str, max_idx: int) -> list[int]:
    """Parse a comma/space-separated string of 1-indexed numbers into 0-indexed ints."""
    result = []
    for token in re.split(r"[,\s]+", raw.strip()):
        try:
            idx = int(token.strip()) - 1
            if 0 <= idx < max_idx:
                result.append(idx)
        except ValueError:
            pass
    return list(dict.fromkeys(result))


def _select_folders(
    doc_registry: dict,
    question: str,
    ollama_client: ollama_lib.Client,
    pageindex_model: str,
) -> set[str]:
    """Ask the LLM to pick relevant top-level folders when the registry is large."""
    folders = sorted({meta["folder"] for meta in doc_registry.values()})
    prompt = (
        "You are a document selector. Given the question below, pick the "
        "most relevant folders from this list.\n\n"
        f"Question: {question}\n\n"
        "Folders:\n" +
        "\n".join(f"{i + 1}. {f}" for i, f in enumerate(folders)) + "\n\n"
        "Reply with a comma-separated list of folder numbers only (e.g. 1,3). "
        "No explanation."
    )
    response = ollama_client.generate(model=pageindex_model, prompt=prompt, think=False)
    indices = _parse_numbers(response.get("response", ""), len(folders))
    if not indices:
        indices = list(range(min(3, len(folders))))
    return {folders[i] for i in indices}


def _coarse_select_docs(
    doc_registry: dict,
    question: str,
    ollama_client: ollama_lib.Client,
    pageindex_model: str,
    pageindex_top_docs: int,
) -> list[str]:
    """
    Stage 1: select the top-N most relevant doc_ids via one (or two) LLM calls.

    If the registry exceeds _MAX_DOCS_IN_PROMPT entries, a folder-level pre-filter
    is applied first so the final doc listing fits in the model's context window.
    """
    doc_ids = list(doc_registry.keys())

    if len(doc_ids) > _MAX_DOCS_IN_PROMPT:
        selected_folders = _select_folders(doc_registry, question, ollama_client, pageindex_model)
        filtered_ids = [
            did for did in doc_ids
            if doc_registry[did]["folder"] in selected_folders
        ][:_MAX_DOCS_IN_PROMPT]
    else:
        filtered_ids = doc_ids

    doc_lines = []
    for i, doc_id in enumerate(filtered_ids):
        meta = doc_registry[doc_id]
        line = f"{i + 1}. {meta['source_path']}"
        if meta.get("description"):
            line += f" — {meta['description']}"
        doc_lines.append(line)

    prompt = (
        f"You are a document selector. Given the question below, pick the "
        f"{pageindex_top_docs} most relevant documents from the list.\n\n"
        f"Question: {question}\n\n"
        "Documents:\n" + "\n".join(doc_lines) + "\n\n"
        "Reply with a comma-separated list of document numbers only (e.g. 1,3,5). "
        "No explanation."
    )
    response = ollama_client.generate(model=pageindex_model, prompt=prompt, think=False)
    indices = _parse_numbers(response.get("response", ""), len(filtered_ids))
    indices = indices[:pageindex_top_docs]
    if not indices:
        indices = list(range(min(pageindex_top_docs, len(filtered_ids))))

    return [filtered_ids[i] for i in indices]


def _retrieve_sections(
    selected_doc_ids: list[str],
    doc_registry: dict,
    question: str,
    ctx: _RetrievalCtx,
) -> list[dict]:
    """
    Stage 2: for each selected document, navigate the PageIndex tree and
    fetch the relevant section text.
    """
    chunks: list[dict] = []
    for doc_id in selected_doc_ids:
        meta = doc_registry[doc_id]
        try:
            structure = ctx.pageindex_client.get_document_structure(doc_id)
            structure_str = (
                json.dumps(structure, indent=2)
                if isinstance(structure, dict)
                else str(structure)
            )
            prompt = (
                f"Given the document structure and the question, identify up to "
                f"{ctx.top_sections} relevant section ranges to retrieve.\n\n"
                f"Question: {question}\n\n"
                f"Document structure:\n{structure_str}\n\n"
                "Reply with page/line ranges as a single comma-separated string "
                "(e.g. '5-7,12' or '3'). If no section is relevant, reply 'none'. "
                "No explanation."
            )
            response = ctx.ollama_client.generate(model=ctx.model, prompt=prompt, think=False)
            pages_str = response.get("response", "").strip()

            if not pages_str or pages_str.lower() == "none":
                logger.info(f"No relevant sections found in {meta['source_path']}")
                continue

            content = ctx.pageindex_client.get_page_content(doc_id, pages=pages_str)
            if not content:
                continue

            chunks.append({
                "text": content,
                "source_path": meta["source_path"],
                "folder": meta["folder"],
                "filename": meta["filename"],
                "chunk_index": 0,
                "score": 1.0,
            })
            logger.info(f"Retrieved '{pages_str}' from {meta['source_path']}")

        except Exception as e:
            logger.warning(f"Failed to retrieve from {meta['source_path']}: {e}")

    return chunks


def pageindex_retrieve(
    question: str,
    doc_registry: dict,
    pageindex_model: str,
    pageindex_workspace: str,
    ollama_base_url: str,
    pageindex_top_docs: int,
    pageindex_top_sections: int,
) -> list[dict]:
    """
    Retrieve relevant content from the PageIndex document registry using
    two-stage LLM-driven navigation.

    Stage 1 (coarse): one Ollama call selects the top-N most relevant
    documents from the registry listing.

    Stage 2 (fine): for each selected document, one Ollama call navigates
    the PageIndex tree structure to identify relevant sections, then
    get_page_content fetches the actual text.

    Args:
        question:              The user's question.
        doc_registry:          Mapping of doc_id -> {source_path, folder,
                               filename, description} from pageindex_index_documents.
        pageindex_model:       Ollama model name used for tree navigation.
        pageindex_workspace:   Directory where PageIndex cached tree structures.
        ollama_base_url:       Ollama server URL.
        pageindex_top_docs:    Max documents to select in stage 1.
        pageindex_top_sections: Max section ranges to fetch per document in stage 2.

    Returns:
        List of chunk dicts with keys: text, source_path, folder,
        filename, chunk_index, score.
    """
    if not doc_registry:
        logger.warning("Empty doc_registry — returning no chunks.")
        return []

    os.environ["OLLAMA_API_BASE"] = ollama_base_url
    ctx = _RetrievalCtx(
        pageindex_client=PageIndexClient(
            model=f"ollama/{pageindex_model}", workspace=pageindex_workspace
        ),
        ollama_client=ollama_lib.Client(host=ollama_base_url),
        model=pageindex_model,
        top_sections=pageindex_top_sections,
    )

    selected_doc_ids = _coarse_select_docs(
        doc_registry, question, ctx.ollama_client, pageindex_model, pageindex_top_docs
    )
    logger.info(
        f"Stage 1 selected {len(selected_doc_ids)} docs: "
        f"{[doc_registry[d]['source_path'] for d in selected_doc_ids]}"
    )

    chunks = _retrieve_sections(selected_doc_ids, doc_registry, question, ctx)

    logger.info(f"pageindex_retrieve: {len(chunks)} chunks for '{question[:60]}'")
    return chunks


def build_prompt(
    question: str,
    chunks: list[dict],
) -> dict:
    """
    Assemble the RAG prompt from the question and retrieved chunks.

    Args:
        question: The user's question.
        chunks:   Retrieved chunks from pageindex_retrieve.

    Returns:
        Dict with keys: system_prompt, user_prompt, question, chunks.
    """
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
    client = ollama_lib.Client(host=ollama_base_url)

    response = client.chat(
        model=llm_model,
        messages=[
            {"role": "system", "content": prompt_data["system_prompt"]},
            {"role": "user", "content": prompt_data["user_prompt"]},
        ],
        think=False,
    )

    answer = response["message"]["content"].strip()

    seen: set[str] = set()
    sources = []
    for chunk in prompt_data["chunks"]:
        src = chunk["source_path"]
        if src not in seen:
            seen.add(src)
            sources.append({"source_path": src, "score": chunk["score"]})

    result = {
        "question": prompt_data["question"],
        "answer": answer,
        "sources": sources,
        "model": llm_model,
    }

    logger.info(f"Answer generated. Sources: {[s['source_path'] for s in sources]}")
    return result
