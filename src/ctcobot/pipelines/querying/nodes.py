"""
Querying pipeline nodes — PageIndex retrieval, prompt assembly, answer generation.
"""
import dataclasses
import logging
import os
import re
from collections import defaultdict

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

    # Collect up to 5 representative filenames per folder so the LLM can see
    # what each folder actually contains rather than guessing from the bare name.
    folder_samples: dict[str, list[str]] = {f: [] for f in folders}
    for meta in doc_registry.values():
        samples = folder_samples[meta["folder"]]
        if len(samples) < 5:
            samples.append(meta["filename"])

    folder_lines = []
    for i, folder in enumerate(folders):
        samples = folder_samples[folder]
        hint = ", ".join(samples) if samples else ""
        folder_lines.append(f"{i + 1}. {folder} [{hint}]")

    prompt = (
        "You are a document selector. Given the question below, pick the "
        "most relevant folders from this list. Each folder shows sample file names.\n\n"
        f"Question: {question}\n\n"
        "Folders (with sample files):\n" +
        "\n".join(folder_lines) + "\n\n"
        "Reply with a comma-separated list of folder numbers only (e.g. 1,3). "
        "No explanation."
    )
    response = ollama_client.generate(
        model=pageindex_model, prompt=prompt, think=False,
        options={"temperature": 0},
    )
    indices = _parse_numbers(response.get("response", ""), len(folders))
    if not indices:
        logger.warning("Folder selection parse failed — using all folders (no filter)")
        return set(folders)
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
    response = ollama_client.generate(
        model=pageindex_model, prompt=prompt, think=False,
        options={"temperature": 0},
    )
    indices = _parse_numbers(response.get("response", ""), len(filtered_ids))
    indices = indices[:pageindex_top_docs]
    if not indices:
        logger.warning("Doc selection parse failed — falling back to first %d docs", pageindex_top_docs)
        indices = list(range(min(pageindex_top_docs, len(filtered_ids))))

    return [filtered_ids[i] for i in indices]


def _flatten_structure(nodes: list[dict], result: list[str] | None = None) -> list[str]:
    """Build a flat list of 'node_id | title | preview' lines from a tree structure."""
    if result is None:
        result = []
    for n in nodes:
        preview = (n.get("text") or "").strip()[:1500].replace("\n", " ")
        result.append(f"{n['node_id']} | {n.get('title', '')} | {preview}")
        _flatten_structure(n.get("nodes", []), result)
    return result


def _collect_node_texts(nodes: list[dict], selected_ids: set[str]) -> list[str]:
    """Return full text of all nodes whose node_id is in selected_ids."""
    texts: list[str] = []
    for n in nodes:
        if n.get("node_id") in selected_ids:
            t = (n.get("text") or "").strip()
            if t:
                texts.append(t)
        texts.extend(_collect_node_texts(n.get("nodes", []), selected_ids))
    return texts


def _collect_node_details(
    nodes: list[dict], selected_ids: set[str]
) -> list[tuple[str, str, str]]:
    """Return (node_id, title, text) for all nodes whose node_id is in selected_ids."""
    results: list[tuple[str, str, str]] = []
    for n in nodes:
        if n.get("node_id") in selected_ids:
            t = (n.get("text") or "").strip()
            if t:
                results.append((n["node_id"], n.get("title", ""), t))
        results.extend(_collect_node_details(n.get("nodes", []), selected_ids))
    return results


def _retrieve_sections(
    selected_doc_ids: list[str],
    doc_registry: dict,
    question: str,
    ctx: _RetrievalCtx,
) -> list[dict]:
    """
    Stage 2: for each selected document, navigate the PageIndex tree using
    text previews and select relevant sections by node_id.

    Each node's first 1500 chars of text are shown to the navigation LLM so it
    can make content-aware decisions rather than guessing from header titles alone.
    The LLM selects node_ids (explicit IDs visible in the prompt) instead of
    line ranges, avoiding ambiguous number parsing.

    Each selected section is emitted as its own chunk so the answer LLM sees
    clearly labelled, distinct excerpts rather than one large concatenated blob.
    """
    chunks: list[dict] = []
    for doc_id in selected_doc_ids:
        meta = doc_registry[doc_id]
        try:
            # Trigger lazy-load: after this call doc['structure'] is in memory with full text
            ctx.pageindex_client.get_document_structure(doc_id)
            structure = ctx.pageindex_client.documents.get(doc_id, {}).get("structure", [])
            if not structure:
                logger.info(f"Empty structure for {meta['source_path']}")
                continue

            nav_lines = _flatten_structure(structure)
            nav_str = "\n".join(nav_lines)

            prompt = (
                "Given the document sections and the question, identify the most relevant sections.\n"
                "Include sections that answer directly OR via acronym expansion, synonyms, or indirect references.\n\n"
                f"Question: {question}\n\n"
                f"Sections (node_id | title | content preview):\n{nav_str}\n\n"
                f"Reply with a comma-separated list of node_id values (e.g. '0001,0003'). "
                f"Select up to {ctx.top_sections} most relevant sections. "
                "If none are relevant, reply 'none'. No explanation."
            )
            response = ctx.ollama_client.generate(
                model=ctx.model, prompt=prompt, think=False,
                options={"temperature": 0},
            )
            node_ids_str = response.get("response", "").strip()

            if not node_ids_str or node_ids_str.lower() == "none":
                logger.info(f"No relevant sections found in {meta['source_path']}")
                continue

            selected_ids = {s.strip() for s in re.split(r"[,\s]+", node_ids_str) if s.strip()}
            node_details = _collect_node_details(structure, selected_ids)
            if not node_details:
                logger.info(f"No content for node_ids {selected_ids} in {meta['source_path']}")
                continue

            for idx, (node_id, title, text) in enumerate(node_details):
                chunks.append({
                    "text": text,
                    "source_path": meta["source_path"],
                    "folder": meta["folder"],
                    "filename": meta["filename"],
                    "chunk_index": idx,
                    "score": 1.0,
                })
            logger.info(f"Retrieved {len(node_details)} section(s) from {meta['source_path']}")

        except Exception as e:
            logger.warning(f"Failed to retrieve from {meta['source_path']}: {e}")

    return chunks


def _rerank_chunks(
    chunks: list[dict],
    question: str,
    ollama_client: ollama_lib.Client,
    model: str,
    top_k: int,
) -> list[dict]:
    """
    Stage 3: rerank all retrieved chunks with a single listwise LLM call,
    then return the top-k most relevant in ranked order.

    Each chunk is shown as a 300-char preview so the reranker prompt stays
    compact. Scores are set to 1/rank so downstream deduplication preserves
    the relevance ordering.
    """
    if len(chunks) <= top_k:
        return chunks

    lines = []
    for i, chunk in enumerate(chunks, start=1):
        preview = chunk["text"].strip()[:600].replace("\n", " ")
        lines.append(f"[{i}] {chunk['source_path']}\n{preview}")

    prompt = (
        "You are a relevance ranker. Given the question and the document excerpts below, "
        "rank the excerpts from most to least relevant for answering the question.\n\n"
        f"Question: {question}\n\n"
        "Excerpts:\n" + "\n\n".join(lines) + "\n\n"
        f"Reply with a comma-separated list of all {len(chunks)} excerpt numbers "
        "in order of relevance (most relevant first), e.g. '3,1,7,2'. No explanation."
    )
    response = ollama_client.generate(
        model=model, prompt=prompt, think=False,
        options={"temperature": 0},
    )
    indices = _parse_numbers(response.get("response", ""), len(chunks))

    # Append any indices the LLM omitted so we never silently drop chunks
    seen = set(indices)
    indices += [i for i in range(len(chunks)) if i not in seen]

    reranked = []
    for rank, i in enumerate(indices[:top_k], start=1):
        chunk = dict(chunks[i])
        chunk["score"] = round(1.0 / rank, 4)
        reranked.append(chunk)

    logger.info(
        "Reranker: %d → %d chunks. Top sources: %s",
        len(chunks), top_k,
        [reranked[j]["source_path"].split("/")[-1] for j in range(min(3, len(reranked)))],
    )
    return reranked


def rewrite_query(
    question: str,
    pageindex_model: str,
    ollama_base_url: str,
) -> str:
    """
    Expand and rewrite the question to improve retrieval recall.

    Expands acronyms, disambiguates generic terms, and adds HR-domain context
    so the coarse and section selectors pick the right documents.
    Falls back to the original question if the LLM returns nothing usable.
    """
    client = ollama_lib.Client(host=ollama_base_url)
    prompt = (
        "You are a search query optimizer for an HR policy knowledge base. "
        "Rewrite the following question to improve document retrieval by: "
        "expanding any acronyms to their full form, adding relevant synonyms, "
        "and making implicit HR topics explicit. "
        "Keep the rewritten query concise (1–2 sentences) and natural. "
        "Output only the rewritten query, nothing else.\n\n"
        f"Question: {question}"
    )
    response = client.generate(
        model=pageindex_model, prompt=prompt, think=False,
        options={"temperature": 0},
    )
    rewritten = response.get("response", "").strip()
    if not rewritten:
        logger.warning("Query rewriting returned empty — using original question")
        return question
    logger.info("Query rewritten: '%s' → '%s'", question[:80], rewritten[:120])
    return rewritten


def pageindex_retrieve(
    question: str,
    expanded_question: str,
    doc_registry: dict,
    pageindex_model: str,
    pageindex_workspace: str,
    ollama_base_url: str,
    pageindex_top_docs: int,
    pageindex_top_sections: int,
    pageindex_reranker_top_k: int,
) -> list[dict]:
    """
    Retrieve relevant content from the PageIndex document registry using
    three-stage LLM-driven navigation.

    Stage 1 (coarse): one Ollama call selects the top-N most relevant
    documents from the registry listing.

    Stage 2 (fine): for each selected document, one Ollama call navigates
    the PageIndex tree structure to identify relevant sections.

    Stage 3 (rerank): one Ollama call reranks all retrieved sections by
    relevance, keeping the top-k most relevant chunks.

    Args:
        question:                  The user's question.
        expanded_question:         Rewritten query used for all retrieval calls.
        doc_registry:              Mapping of doc_id -> {source_path, folder,
                                   filename, description}.
        pageindex_model:           Ollama model name used for navigation.
        pageindex_workspace:       Directory where PageIndex caches tree structures.
        ollama_base_url:           Ollama server URL.
        pageindex_top_docs:        Max documents to select in stage 1.
        pageindex_top_sections:    Max section ranges to fetch per document in stage 2.
        pageindex_reranker_top_k:  Max chunks to keep after stage 3 reranking.

    Returns:
        List of chunk dicts with keys: text, source_path, folder,
        filename, chunk_index, score (1/rank after reranking).
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
        doc_registry, expanded_question, ctx.ollama_client, pageindex_model, pageindex_top_docs
    )
    logger.info(
        "Stage 1 selected %d docs: %s",
        len(selected_doc_ids),
        [doc_registry[d]["source_path"] for d in selected_doc_ids],
    )

    chunks = _retrieve_sections(selected_doc_ids, doc_registry, expanded_question, ctx)

    if not chunks:
        logger.warning(
            "pageindex_retrieve: no chunks retrieved for '%s' — answer will have no context",
            question[:60],
        )
        return chunks

    # Stage 3: rerank with 3× candidates so the post-rank diversity filter has enough to work with
    pre_div_k = min(len(chunks), pageindex_reranker_top_k * 3)
    ranked = _rerank_chunks(chunks, expanded_question, ctx.ollama_client, pageindex_model, pre_div_k)

    # Post-rerank diversity: keep top-2 per source while preserving relevance order
    source_count: dict[str, int] = defaultdict(int)
    final: list[dict] = []
    for chunk in ranked:
        if source_count[chunk["source_path"]] < 2:
            final.append(chunk)
            source_count[chunk["source_path"]] += 1
        if len(final) >= pageindex_reranker_top_k:
            break
    chunks = final or ranked[:pageindex_reranker_top_k]

    logger.info("pageindex_retrieve: %d chunks after reranking+diversity for '%s'", len(chunks), question[:60])
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
