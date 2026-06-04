"""
Indexing pipeline nodes — T-07: ingest_documents
"""
import json
import logging
from pathlib import Path
import re
import frontmatter
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
import ollama
import tiktoken
from turbovec import TurboQuantIndex

logger = logging.getLogger(__name__)


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

            # Relative path from corpus root (used as source identifier)
            relative_path = str(file_path.relative_to(root))

            # Top-level folder (e.g. "legal", "people-group")
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

def clean_documents(
    documents: list[dict],
    min_doc_tokens: int,
    encoding_name: str,
) -> list[dict]:
    """
    Clean raw documents: strip YAML frontmatter, normalize whitespace,
    and filter out documents that are too short to be useful.

    Args:
        documents:      Output of ingest_documents.
        min_doc_tokens: Minimum token count to keep a document.
        encoding_name:  tiktoken encoding name for token counting.

    Returns:
        List of cleaned document dicts with keys:
        path, text, folder, filename, token_count.
    """
    import tiktoken
    enc = tiktoken.get_encoding(encoding_name)

    cleaned = []
    filtered_short = 0
    filtered_empty = 0

    for doc in documents:
        raw = doc["raw_text"]

       # Pre-clean: remove degenerate lone --- lines before frontmatter parsing
        # e.g. "---\n\n---\n\n## content" confuses python-frontmatter
        pre = re.sub(r'^(---\s*\n\s*\n)+', '', raw.strip())

        # Strip YAML frontmatter
        pre = re.sub(r'^(---\s*\n\s*\n)+', '', raw.strip())
        try:
            parsed = frontmatter.loads(pre)
            text = parsed.content.strip()
        except Exception:
            text = pre.strip()

        # Remove Hugo shortcodes FIRST before sweeping orphaned ---
        text = re.sub(r'\{\{[%<].*?[%>]\}\}', '', text, flags=re.DOTALL)

        # Final sweep: strip any leftover lone --- lines at the top
        lines = text.split('\n')
        while lines and re.match(r'^---\s*$', lines[0]):
            lines.pop(0)
            while lines and not lines[0].strip():
                lines.pop(0)
        text = '\n'.join(lines).strip()

        # Normalize excessive whitespace (3+ newlines → 2)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()

        # Normalize excessive whitespace (3+ newlines → 2)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()

        if not text:
            filtered_empty += 1
            continue

        token_count = len(enc.encode(text))

        if token_count < min_doc_tokens:
            filtered_short += 1
            continue

        cleaned.append({
            "path": doc["path"],
            "text": text,
            "folder": doc["folder"],
            "filename": doc["filename"],
            "token_count": token_count,
        })

    logger.info(
        f"Cleaning complete. Kept: {len(cleaned)} | "
        f"Filtered (too short): {filtered_short} | "
        f"Filtered (empty): {filtered_empty}"
    )
    return cleaned

def chunk_documents(
    documents: list[dict],
    chunk_size: int,
    chunk_overlap: int,
    encoding_name: str,
) -> list[dict]:
    """
    Split cleaned documents into fixed-size token chunks with overlap
    using LangChain's RecursiveCharacterTextSplitter with tiktoken backend.

    Args:
        documents:      Output of clean_documents.
        chunk_size:     Maximum tokens per chunk.
        chunk_overlap:  Number of tokens to overlap between chunks.
        encoding_name:  tiktoken encoding name.

    Returns:
        List of chunk dicts with keys:
        chunk_id, source_path, folder, filename, text, token_count,
        chunk_index, total_chunks.
    """

    enc = tiktoken.get_encoding(encoding_name)

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=encoding_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = []

    for doc in documents:
        split_texts = splitter.split_text(doc["text"])

        if not split_texts:
            continue

        doc_chunks = []
        for chunk_index, chunk_text in enumerate(split_texts):
            token_count = len(enc.encode(chunk_text))
            doc_chunks.append({
                "chunk_id": f"{doc['path']}::chunk_{chunk_index}",
                "source_path": doc["path"],
                "folder": doc["folder"],
                "filename": doc["filename"],
                "text": chunk_text,
                "token_count": token_count,
                "chunk_index": chunk_index,
                "total_chunks": len(split_texts),
            })

        chunks.extend(doc_chunks)

    logger.info(
        f"Chunking complete. {len(documents)} docs → {len(chunks)} chunks "
        f"(chunk_size={chunk_size}, overlap={chunk_overlap})"
    )
    return chunks

def embed_and_index(
    chunks: list[dict],
    ollama_base_url: str,
    embedding_model: str,
    turbovec_persist_path: str,
) -> dict:
    """
    Embed all chunks using Ollama and build a turbovec TurboQuantIndex.

    Saves two files into turbovec_persist_path:
      - index.tq   — turbovec binary (quantized vectors, row i = chunk i)
      - meta.json  — parallel chunk metadata + index config

    Args:
        chunks:                Output of chunk_documents.
        ollama_base_url:       Ollama server URL.
        embedding_model:       Ollama embedding model name.
        turbovec_persist_path: Directory to persist the index and metadata.

    Returns:
        Dict with indexing summary stats.
    """
    Path(turbovec_persist_path).mkdir(parents=True, exist_ok=True)
    index_path = str(Path(turbovec_persist_path) / "index.tq")
    meta_path = str(Path(turbovec_persist_path) / "meta.json")

    ollama_client = ollama.Client(host=ollama_base_url)

    BATCH_SIZE = 50
    total = len(chunks)
    indexed = 0
    errors = 0
    all_embeddings: list[np.ndarray] = []
    all_meta: list[dict] = []

    for batch_start in range(0, total, BATCH_SIZE):
        batch = chunks[batch_start: batch_start + BATCH_SIZE]

        for chunk in batch:
            try:
                response = ollama_client.embeddings(
                    model=embedding_model,
                    prompt=chunk["text"],
                )
                emb = np.array(response["embedding"], dtype=np.float32)
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm
                all_embeddings.append(emb)
                all_meta.append({
                    "text": chunk["text"],
                    "source_path": chunk["source_path"],
                    "folder": chunk["folder"],
                    "filename": chunk["filename"],
                    "chunk_index": chunk["chunk_index"],
                })
                indexed += 1
            except Exception as e:
                logger.warning(f"Failed to embed chunk {chunk['chunk_id']}: {e}")
                errors += 1

        if batch_start % 500 == 0:
            logger.info(f"Progress: {batch_start}/{total} chunks embedded...")

    vectors = np.stack(all_embeddings)  # shape (indexed, dim)
    dim = vectors.shape[1]
    index = TurboQuantIndex(dim=dim, bit_width=4)
    index.add(vectors)
    index.write(index_path)

    with open(meta_path, "w") as f:
        json.dump({"dim": dim, "bit_width": 4, "chunks": all_meta}, f)

    summary = {
        "total_chunks": total,
        "indexed": indexed,
        "errors": errors,
        "turbovec_path": turbovec_persist_path,
    }
    logger.info(f"Indexing complete: {summary}")
    return summary