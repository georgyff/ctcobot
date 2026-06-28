# CTCoBot

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

An on-premises HR-policy Q&A assistant over a company handbook. It runs a fully
local, agentic RAG pipeline (HyDE vector retrieval + folder-scoped BM25, LLM
reranking and answer synthesis) on top of [Ollama](https://ollama.com)

---

## 1. Prerequisites

| Requirement | Notes |
|-------------|-------|
| Python 3.10+ | Requires 3.10 or newer (developed on 3.12). |
| [Ollama](https://ollama.com) | Must be installed and running locally (or reachable over the network). |
| Ollama models | Pull the models you intend to use for the pipeline. |
| Disk | Enough for the corpus + the quantized index (`data/04_feature/turbovec_db`). |
| RAM / GPU | Depends on the model you're running, but for most, a GPU is recommended but not required. |

### Pull the models 

```bash
ollama pull nomic-embed-text     # embeddings
ollama pull qwen3.5:4b           # answer LLM / router / reranker / judge
```

Or whichever local models you prefer.

Confirm Ollama is up:

```bash
curl http://localhost:11434/api/version
```

---

## 2. Install

```bash
# from the project root — installs the deps AND the `ctcobot` command
pip install -e .
# or, with uv:
uv sync
```

This installs the `ctcobot` CLI (`ctcobot ask`, `ctcobot index`, `ctcobot evaluate`).

---

## 3. Deploy on company dataset

### 3.1 Add the new corpus

Documents are ingested from a directory tree of **Markdown files**. The
**top-level folder of each file is used as its routing category**, so lay the
corpus out as:

```
data/01_raw/<your_company>_handbook/
├── benefits/        ← top-level folder = routing category
│   └── ...md
├── legal/
│   └── ...md
└── people/
    └── ...md
```

> Only `*.md` is ingested — `.pdf`, `.docx`, `.html`, `.txt` are silently
> skipped. Convert other formats to Markdown first. Hugo/YAML frontmatter and
> Hugo shortcodes are stripped automatically; other templating is not.

### 3.2 Point the config at the new corpus

Edit `conf/base/parameters.yml`:

```yaml
raw_data_path: "data/01_raw/<company_name>_handbook"   # was gitlab_handbook

# Folders that are ALWAYS included in the retrieval filter (a safety floor so
# HR/policy folders survive router variance). Replace with your top-level
# folder names:
#If your handbook contains ONLY HR policy documents, leave empty
priority_folders:
  - "legal"
  - "people"
  - "benefits"
```

### 3.3 Adapt the routing/answer prompts

`src/ctcobot/prompt_templates.py` contains company neutral prompts for various LLM calls, edit to get better results or leave unchanged for a baseline.

### 3.4 Configure the Ollama endpoint

Defaults to `http://localhost:11434`. To point at a remote/host Ollama, set
`ollama_base_url` in `conf/base/parameters.yml`.

To override it per-environment without editing the base config, set
`ollama_base_url` in `conf/local/parameters.yml` instead — `conf/local/` is
gitignored and is merged over `conf/base/` by Kedro's config loader (see
`src/ctcobot/settings.py`), so keep environment-specific values there.

### 3.5 Build the index

```bash
ctcobot index
```

This runs ingest → clean → chunk → embed and writes the quantized vector index to
`data/04_feature/turbovec_db/` (`index.tq` + `meta.json`). Re-run whenever the
corpus changes (there is no incremental indexing at the moment — it reprocesses everything).

Check `data/07_model_output/indexing_summary.json` for the chunk count and that
errors are ~0.

### 3.6 Ask a question

```bash
ctcobot ask -q "What is the remote work policy?"
```

It prints the grounded answer, the tools used, and the source files cited — verify
the sources point at your corpus.

---

## 4. Evaluate

The benchmark scores retrieval and answer quality against a CSV of question /
expected-answer / source-file rows: `data/05_model_input/eval_qa_pairs_blank.csv`
(wired in `conf/base/catalog.yml` as `eval_qa_pairs`).

The benchmark currently only contains the questions. To use it with your company handbook populate the "expected_answer","source_file" and "folder" fields with the appropriate data either via an LLM or manually.

If the answer to one of the general questions isn't in the handbook, set the expected answer to "Answer not found in the handbook" and leave a `-` in other fields. Thereby if the LLM doesn't hallucinate a response, the question is scored a 5 on quality and is excluded from the retrieval metric averages (hit rate, MRR, precision, recall); it still counts toward latency.

```bash
ctcobot evaluate          # writes data/08_reporting/benchmark_report*.json
```

Acceptance targets are externalized to parameters.yml
---

## 5. How the pipelines work

The project is built on [Kedro](https://kedro.org). Four pipelines are
registered in `src/ctcobot/pipeline_registry.py`; the `ctcobot` CLI is a thin
wrapper that runs them (except `ask`, which drives the agent directly).

| Pipeline | CLI command | Kedro pipeline | What it produces |
|----------|-------------|----------------|------------------|
| Indexing | `ctcobot index` | `indexing` | The quantized vector index (`index.tq` + `meta.json`). |
| Querying (interactive) | `ctcobot ask` | *(agent, not Kedro)* | A grounded answer + sources for one question. |
| Evaluation | `ctcobot evaluate` | `evaluation` | The benchmark report JSON. |
| Graph indexing | `ctcobot index_graph` | `lightrag_indexing` | An optional LightRAG knowledge-graph store. |

A standalone linear `querying` Kedro pipeline also exists (see 5.2) but the CLI
does not use it.

### 5.1 Indexing pipeline (`ctcobot index` → `indexing`)

Turns the Markdown corpus into a searchable, quantized vector index. Four nodes
run in sequence (`src/ctcobot/pipelines/indexing/nodes.py`):

1. **`ingest_documents`** — recursively reads every `*.md` file under
   `raw_data_path`. For each file it records the relative path, raw text, the
   **top-level folder** (used as the routing category), and the filename. Empty
   or unreadable files are skipped. → `raw_docs`
2. **`clean_documents`** — strips YAML frontmatter and Hugo shortcodes,
   collapses excess whitespace, counts tokens with `encoding_name`, and drops
   documents shorter than `min_doc_tokens`. → `cleaned_docs`
3. **`chunk_documents`** — splits each cleaned document into `chunk_size`-token
   chunks with `chunk_overlap` overlap, using LangChain's
   `RecursiveCharacterTextSplitter` on the tiktoken encoder. → `chunked_docs`
4. **`embed_and_index`** — embeds every chunk with the Ollama `embedding_model`,
   L2-normalizes the vectors, builds a 4-bit-quantized `TurboQuantIndex`, and
   writes `index.tq` (vectors) + `meta.json` (parallel chunk metadata: text,
   source_path, folder, filename, chunk_index) into `turbovec_persist_path`.
   A summary lands in `data/07_model_output/indexing_summary.json`.

The same `meta.json` is the single source of chunk text and metadata for **both**
vector and BM25 retrieval at query time, so the two searches always cover an
identical set of chunks.

### 5.2 Querying — the agentic RAG path (`ctcobot ask`)

`ctcobot ask` does **not** run a Kedro pipeline. It instantiates a
`VectorRAGTool`, a `KeywordRAGTool`, and a `QueryAgent`
(`src/ctcobot/pipelines/querying/{tools,agent}.py`) and answers one question:

1. **Folder ranking** — one LLM call (`rank_folders`) ranks the corpus's
   top-level folders by relevance to the question. The result is computed once
   and shared by both tools (no duplicate call). Downstream it is always unioned
   with the `priority_folders` floor so canonical HR folders survive router
   variance.
2. **Routing** — the tool-calling `agent_model` decides whether to also use
   keyword search. `vector_rag` **always** runs as a semantic floor; the router
   (plus an acronym / quoted-phrase / hyphenated-term safety net) only decides
   whether to **also** run `keyword_rag`. This prevents keyword-only misses.
3. **`vector_rag` (HyDE)** — the LLM writes a hypothetical answer paragraph
   (HyDE), that paragraph is embedded, and `retrieve_chunks_folder_priority`
   oversamples `top_k × retrieve_oversample` candidates from the index, filters
   to the top `top_folders` ranked folders plus the `priority_folders` floor,
   keeps the top `top_k`, and reranks those with `reranker_model` before
   returning.
4. **`keyword_rag` (BM25)** — folder-scoped BM25 over the same chunks (a cached
   index built from `meta.json`), restricted to the same folder set, returning
   `bm25_top_k` candidates. Targets exact-term / acronym questions embeddings
   blur.
5. **Merge + rerank** — the vector tool returns candidates already reranked by
   `reranker_model`; the keyword tool returns its BM25 candidates unreranked.
   The agent merges and deduplicates both by `(source_path, chunk_index)`, then
   reranks the merged set **again** with `agent_reranker_model` (a listwise LLM
   rerank) down to `rerank_top_n` chunks.
6. **Answer** — `generate_answer` calls `llm_model` (temperature 0.2) on the
   reranked chunks under the grounding `SYSTEM_PROMPT`, returning the answer plus
   deduplicated source citations.

**Legacy linear `querying` pipeline.** The `querying` Kedro pipeline
(`pipelines/querying/pipeline.py`) wires the same retrieval nodes — `generate_hyde_doc`
→ `embed_query` → `rank_folders` → `retrieve_chunks_folder_priority` →
`rerank_chunks` → `build_prompt` → `generate_answer` — linearly for a single
`params:question`, with no agent, no BM25, and no merge step. It is runnable with
`kedro run --pipeline querying` but is **not** used by the CLI; the agentic path
above superseded it.

### 5.3 Evaluation pipeline (`ctcobot evaluate` → `evaluation`)

Benchmarks the agentic path against the `eval_qa_pairs` CSV. Five nodes
(`src/ctcobot/pipelines/evaluation/nodes.py`):

1. **`run_eval_pipeline`** — runs the **same `QueryAgent`** (5.2) once per CSV
   row, recording the answer, sources, tools used, the deduped pre-rerank
   candidate set (for retrieval metrics), and end-to-end latency. → `qa_eval_results`
2. **`compute_retrieval_metrics`** — over a fixed, deduplicated top-`retrieval_eval_k`
   candidate window, computes **hit rate**, **MRR**, **precision**, and
   **recall** against each row's expected `source_file`. Out-of-corpus rows
   (blank / `-` / `none` / `n/a`) are excluded so they don't distort retrieval.
3. **`compute_quality_metrics`** — an LLM judge (`judge_model`, temperature 0,
   JSON-constrained, with parse-salvage + one retry) scores each answer **1–5**
   against the expected answer. A correct out-of-corpus refusal is floored to 5
   (`refusal_floor_applied`); an unparseable judgment scores 0 and is excluded
   from the average.
4. **`compute_latency_metrics`** — p50 / p95 / p99 / avg from the per-question
   timings captured in step 1 (no separate latency loop).
5. **`save_benchmark_report`** — aggregates everything, applies the
   `benchmark_targets` thresholds to set each `*_pass` flag, prints the console
   summary table, and writes
   `data/08_reporting/benchmark_report_v<benchmark_version>.json` (dots → dashes).

### 5.4 Graph indexing pipeline (`ctcobot index_graph` → `lightrag_indexing`)

**Optional / experimental** — not wired into the answer path. A single node
`build_lightrag_index` builds a [LightRAG](https://github.com/HKUDS/LightRAG)
knowledge graph from the `cleaned_docs` produced by the indexing pipeline,
filtered to `lightrag_index_folders`, using `llm_model` for entity/relationship
extraction and `embedding_model` for the graph's vector components. The store is
written to `lightrag_working_dir`. Because it consumes `cleaned_docs`, run
`ctcobot index` first. It is slow (many LLM calls) and currently informational
only.

---

## 6. Tuning knobs (`conf/base/parameters.yml`)

| Param | Default | Purpose |
|-------|---------|---------|
| `chunk_size` / `chunk_overlap` | 256 / 50 | Token chunking granularity. |
| `top_k` | 10 | Vector candidates before rerank. |
| `top_folders` | 5 | Folders the router restricts retrieval to. |
| `rerank_top_n` | 6 | Chunks kept after rerank → answer context. |
| `bm25_top_k` | 10 | Keyword (BM25) candidates before merge. |
| `min_doc_tokens` | 50 | Drop documents shorter than this. |
| `embedding_model` / `llm_model` / `reranker_model` / `agent_model` / `judge_model` | see file | Ollama model names. |

---

## 7. Notes & current limitations

- **Interface is the CLI only.** There is no HTTP API or web UI yet — integrate by
  calling `QueryAgent.run()` (see `src/ctcobot/pipelines/querying/agent.py`) or
  wrapping the CLI.
- **Privacy:** all inference is local via Ollama; the pipeline makes no external
  API calls.