# CTCoBot

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

An on-premises HR-policy Q&A assistant over a company handbook. It runs a fully
local, agentic RAG pipeline (HyDE vector retrieval + folder-scoped BM25, LLM
reranking and answer synthesis) on top of [Ollama](https://ollama.com)

---

## 1. Prerequisites

| Requirement | Notes |
|-------------|-------|
| Python 3.12 | Project targets 3.12. |
| [Ollama](https://ollama.com) | Must be installed and running locally (or reachable over the network). |
| Ollama models | Pull the models you intend to use for the pipeline. |
| Disk | Enough for the corpus + the quantized index (`data/04_feature/turbovec_db`). |
| RAM / GPU | Depends on the model you're running, but for most, a GPU is recommended but not required. |

### Pull the models 

```bash
ollama pull nomic-embed-text     # embeddings
ollama pull qwen3.5:4b           # answer LLM / router / reranker / judge
ollama pull qwen3.5:0.8b         # lightweight per-tool reranker
```

Or whichever local models you prefer.

Confirm Ollama is up:

```bash
curl http://localhost:11434/api/version
```

---

## 2. Install

```bash
# from the project root
pip install -r requirements.txt
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
#If your handbook only contains HR policies, leave empty
priority_folders:
  - "legal"
  - "people"
  - "benefits"
```

### 3.3 Adapt the routing/answer prompts

`src/ctcobot/prompt_templates.py` contains GitLab-specific domain knowledge that
must be rewritten for the new handbook:

- **`FOLDER_RANK_SYSTEM_PROMPT`** — the most coupled block. It tells the router
  which folder holds which topic (and references GitLab terms like *9-box*,
  *TNTR*). Rewrite its folder names and topic→folder hints for your taxonomy.
  (The folder *list* is discovered from the index automatically; only the
  guidance text needs editing.)
- **`SYSTEM_PROMPT`** — remove GitLab system names (Tilt, Workday, EthicsPoint,
  Lighthouse, the CPO/CLO reporting paths). The generic grounding and
  answer-directness rules are portable; keep them.
- **`AGENT_SYSTEM_PROMPT`** — the acronym examples (`TNTR, FMLA, EEO, RSU,
  9-box`) are illustrative; swap for your domain's salient acronyms or
  generalize.

### 3.4 Configure the Ollama endpoint

Defaults to `http://localhost:11434`. To point at a remote/host Ollama, set it in
**both**:

- `conf/base/parameters.yml` → `ollama_base_url`
- `conf/local/credentials.yml` → `ollama.base_url`

(`conf/local/` is gitignored — keep environment-specific values there.)

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

## 4. Evaluate (optional but recommended)

The benchmark scores retrieval and answer quality against a CSV of question /
expected-answer / source-file rows: `data/05_model_input/eval_qa_pairs_general.csv`
(wired in `conf/base/catalog.yml` as `eval_qa_pairs`).

To benchmark a new deployment, **re-ground this CSV to your handbook**: keep the
general HR questions, but rewrite each `expected_answer` and `source_file` to match
your corpus (or set `source_file` to `-` and the expected answer to a "not found"
statement where your handbook is silent — those rows are scored on correct
refusal, not retrieval). Then:

```bash
ctcobot evaluate          # writes data/08_reporting/benchmark_report*.json
```

Acceptance targets (corpus-independent): hit_rate ≥ 0.70, MRR ≥ 0.55,
avg_quality ≥ 3.5, recall ≥ 0.35, precision ≥ 0.15.

---

## 5. Tuning knobs (`conf/base/parameters.yml`)

| Param | Default | Purpose |
|-------|---------|---------|
| `chunk_size` / `chunk_overlap` | 256 / 50 | Token chunking granularity. |
| `top_k` | 10 | Vector candidates before rerank. |
| `top_folders` | 7 | Folders the router restricts retrieval to. |
| `rerank_top_n` | 6 | Chunks kept after rerank → answer context. |
| `bm25_top_k` | 10 | Keyword (BM25) candidates before merge. |
| `min_doc_tokens` | 50 | Drop documents shorter than this. |
| `embedding_model` / `llm_model` / `reranker_model` / `agent_model` / `judge_model` | see file | Ollama model names. |

---

## 6. Notes & current limitations

- **Interface is the CLI only.** There is no HTTP API or web UI yet — integrate by
  calling `QueryAgent.run()` (see `src/ctcobot/pipelines/querying/agent.py`) or
  wrapping the CLI.
- **Docker is a development container** (`Dockerfile` / `docker-compose.yaml` run
  JupyterLab on port 8888) — not a production runtime. Ollama must run as a
  separate service.
- **Telemetry:** this is a Kedro project; disable usage telemetry for a private
  deployment with `kedro telemetry off` (or remove `[tool.kedro_telemetry]` from
  `pyproject.toml`).
- **Privacy:** all inference is local via Ollama; the pipeline makes no external
  API calls.

---

## 7. Running tests

```bash
pytest
```

Coverage thresholds are configured in `pyproject.toml` under
`[tool.coverage.report]`. Note current test coverage is minimal (Kedro smoke
tests).
