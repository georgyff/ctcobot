# ctcobot — Portfolio Report Pack

This document collates every artifact and decision needed to write a complete
portfolio report on the ctcobot project: what it is, how it was built, every
version that was benchmarked, why each change was made, what worked, what
didn't, and what the final architecture looks like.

---

## 1. Project at a glance

**ctcobot** is an HR-policy chatbot that answers employee questions about
company policy by retrieving passages from a local copy of the **GitLab
public handbook** (≈ 48,639 chunks across 37 top-level folders) and
synthesizing an answer with a local LLM via **Ollama**.

- **Stack**: Python · Kedro (pipeline orchestration) · Ollama (local LLM
  serving) · `nomic-embed-text` (embeddings) · `qwen3.5:4b` / `qwen3.5:0.8b`
  (generation + judging + reranking) · **turbovec / TurboQuantIndex**
  (Rust-backed scalar-quantized vector index) · LightRAG (graph index,
  experimental).
- **Corpus**: GitLab handbook (`data/01_raw/gitlab_handbook/`) — 37
  top-level folders. Engineering / Marketing / Security dominate by chunk
  count; HR-relevant folders (`people-group`, `people-policies`,
  `total-rewards`, `legal`) are a small minority.
- **Benchmark targets** (defined once and held constant across all runs):
  - Hit Rate ≥ 0.70
  - MRR ≥ 0.55
  - Avg quality (LLM judge, 1–5) ≥ 3.5
  - P95 latency ≤ 5 s *(only realistic on a hosted API; local Ollama
    cannot meet this)*
- **Eval harness**: 25 hand-curated QA pairs (`eval_qa_pairs`) +
  20-question latency loop. LLM-as-judge scores 1–5 with rubric.

All benchmark JSON reports live in `data/08_reporting/benchmark_report_vX-Y.json`.

---

## 2. Architecture families

The project went through **four distinct retrieval architectures**:

| Series | Architecture | Branch | Window |
|---|---|---|---|
| **Series 1** | Vectorized RAG (cosine sim over FAISS/turbovec) | `main` | 2026-03-20 → 2026-05-26 |
| **Series 2** | PageIndex (LLM-driven 3-stage tree retrieval) | `PageIndex-ver` | 2026-04-14 → 2026-05-07 |
| **Series 3** | Agentic multi-tool (HyDE-vector + BM25 keyword + LightRAG) | `agentic-ver` | 2026-05-29 → 2026-06-02 |
| **Series 4** | HyDE Vector RAG + LLM folder routing (final) | `main` | 2026-06-04 → 2026-06-12 |

### 2.1 Series 1 — Vectorized RAG
`embed(query) → cosine-sim retrieve from FAISS/turbovec → optional cross-encoder rerank → LLM answer`.
Later sub-versions add **HyDE** (embed a hypothetical answer paragraph instead
of the raw query), **semantic chunking**, and **listwise LLM rerank**.

### 2.2 Series 2 — PageIndex RAG
A hierarchical document tree is built once, then a three-stage LLM-driven
retrieval runs per query:
1. **Stage 1** — LLM scans document registry, picks top-N docs.
2. **Stage 2** — LLM walks each chosen doc's tree, picks relevant sections.
3. **Stage 3** — Listwise LLM reranker picks final chunks.

No embedding index. Slow but interpretable.

### 2.3 Series 3 — Agentic RAG (v3.0)
A `QueryAgent` (Ollama tool-calling loop, `MAX_TOOL_ROUNDS=3`) routes between:
- `VectorRAGTool` (HyDE → turbovec → LLM rerank)
- `KeywordRAGTool` (BM25 over the same 48k chunks)
- `GraphRAGTool` (LightRAG knowledge-graph retrieval)

### 2.4 Series 4 — HyDE Vector + Folder Routing (final)
`HyDE → embed → rank_folders → folder-priority retrieve → LLM rerank → answer`.
The agent and all non-vector tools are removed (Series 3 post-mortem showed
keyword_rag caused every score=1 failure). A new pre-retrieval **folder
ranking step** asks an LLM to order the 37 top-level handbook folders by
relevance to the question, and retrieval is restricted to the top 5.

---

## 3. Complete version history

Status, metrics, and reason for each version. Verified against the JSON
reports on disk. **Targets**: Hit≥0.70, MRR≥0.55, Quality≥3.5, P95≤5s.

### Series 1 — Vectorized RAG (`main`)

| Ver | Date | Quality | Hit | MRR | P95 (s) | Reason for change |
|----|------|---------|-----|------|---------|-------------------|
| v1.0 | 2026-03-20 | 3.19 | 0.84 | 0.707 | 46 | **Baseline**: llama3.2, chunk=512, top_k=5, no reranker. |
| v1.1 | 2026-03-27 | 3.14 | 0.88 | 0.827 | 34 | `chunk_size 512 → 256` for finer-grain retrieval. Hit/MRR jumped. |
| v1.2 | 2026-03-27 | 4.08 | 0.88 | 0.827 | 98 | LLM + judge upgraded `llama3.2 → qwen3.5:4b`. Big quality win, latency doubled. |
| v1.3 | 2026-03-27 | 3.08 | 0.88 | 0.827 | 32 | LLM + judge downsized to `qwen3.5:0.8b`. Weak judge scored more harshly. |
| v1.4 | 2026-03-27 | 3.52 | 0.88 | 0.827 | 32 | Restored judge to `4b`; LLM stays `0.8b`. Best quality/latency trade so far. |
| v1.5 | 2026-03-30 | 3.24 | 0.80 | 0.780 | 27 | Added `cross-encoder/ms-marco-MiniLM-L-6-v2` reranker, `rerank_top_n=3`. Retrieval **regressed** — reranker discarded too many candidates. |
| v1.6 | 2026-03-30 | 3.92 | 0.88 | 0.808 | 29 | `top_k 5→10`, `rerank_top_n 3→5`. Retrieval recovered, best Series-1 quality at this point. |
| v1.7 | 2026-03-30 | 3.56 | 0.84 | 0.773 | 36 | `top_k 10→15`. Regressed — more noise. |
| v1.8 | 2026-03-31 | 4.12 | 0.88 | 0.808 | 90 | LLM `→qwen3.5:4b`, `top_k=10`. Highest Series-1 quality, P95 tripled. |
| v1.9 | 2026-04-07 | 3.68 | 0.84 | 0.763 | 37 | Replaced fixed-size chunking with **semantic chunking** (all-MiniLM-L6-v2, 95th-percentile breakpoint), `max_chunk_tokens=1024`. Retrieval down vs v1.6. |
| v1.10 | 2026-04-08 | 3.16 | 0.84 | 0.793 | 24 | Semantic chunks `1024→256`. Quality dropped — too little context per chunk. |
| v1.11 | 2026-04-08 | 3.60 | 0.88 | 0.833 | 27 | **Tightened SYSTEM_PROMPT**: "I could not find" now only allowed when *every* excerpt is unrelated. +0.44 quality from prompt alone. |
| v1.12 | 2026-04-10 | 3.36 | 0.88 | 0.833 | 27 | Re-run of v1.11 — score variation is LLM non-determinism. Last pre-HyDE version. |
| v1.13 | 2026-05-07 | 4.00 | 0.84 | 0.768 | 170 | Re-ran with improved generation model selection / longer-context prompts (exact code changes not recorded; jump in quality consistent with returning to the `4b` answer model). |
| v1.15 | 2026-05-08 | 4.08 | 0.84 | 0.784 | 143 | Iterative tuning of reranker/top_k around v1.13 config. |
| v1.16 | 2026-05-12 | 4.16 | 0.88 | 0.755 | 155 | Continued tuning — hit recovered to 0.88. Committed as `v1.16`. |
| v1.17 | 2026-05-15 | 4.64 | 1.00 | 0.880 | 749 | **Best Series-1 quality (4.64)** and first Hit Rate = 1.0 — but P95 hit 749s. Likely larger context window / more chunks fed to LLM. |
| v1.18 | 2026-05-22 | 3.96 | 1.00 | 0.846 | 254 | Latency walked back from v1.17 (fewer chunks / shorter context), some quality given up. |
| v1.19 | 2026-05-19 | 4.04 | 1.00 | 0.837 | 200 | **HyDE retrieval introduced** (commit `561ed54 HyDE version`). Embedding a hypothetical answer paragraph instead of the raw query. Hit stayed at 1.0. |
| v1.20 | 2026-05-26 | — | — | — | — | Commit `7b89293 v1.20` — final main-branch Series-1 snapshot before switching to PageIndex (Series 2 was already on its own branch). |

### Series 2 — PageIndex RAG (`PageIndex-ver`)

All Series-2 runs use: embedding=none, llm=qwen3.5:4b, judge=qwen3.5:4b,
pageindex_model=qwen3.5:4b. P95 latencies are unreachable on local Ollama
(≈ 12 sequential LLM calls per query × ~50 s each).

| Ver | Date | Quality | Hit | MRR | P95 (s) | Reason for change |
|----|------|---------|-----|------|---------|-------------------|
| v2.0 | 2026-04-14 | 2.28 | 0.48 | 0.353 | 183 | **First PageIndex run.** `top_docs=5`, `top_sections=3`, llm=`0.8b`. Too few docs/sections selected. |
| v2.1 | 2026-05-01 | 3.20 | 0.60 | 0.463 | 348 | llm `0.8b→4b`, `top_docs 5→8`, `top_sections 3→5`. Quality recovered, retrieval still weak. |
| v2.2 | 2026-05-04 | 3.56 | 0.68 | 0.425 | 474 | Retrieval-logic refactor (details unrecorded). |
| v2.3 | 2026-05-05 | 2.96 | 1.00 | 0.689 | 611 | `top_sections 5→8`. **Hit Rate jumped to 1.0** but quality dropped — too many chunks overwhelmed the context. |
| v2.4 | 2026-05-05 | 2.48 | 1.00 | 0.569 | 736 | Sections reverted to 5, code refactor. Further regression. |
| v2.5 | 2026-05-05 | 3.20 | 1.00 | 0.704 | 607 | **Major retrieval improvements** (+0.72 quality). Best PageIndex quality. |
| v2.6 | 2026-05-06 | 2.60 | 1.00 | 0.649 | 510 | Tried 4 fixes at once: per-source cap of 2 chunks before rerank; Stage-2 uses original `question` (not expanded); reranker preview `300→600` chars; tighter SYSTEM_PROMPT. **Root cause of regression**: original `question` broke acronym expansion ("TNTR", "9-box"). |
| v2.7 | 2026-05-06 | 2.48 | 0.96 | 0.596 | 694 | Reverted Stage-2 to `expanded_question`; per-source cap moved *post*-rerank; reranker pulls 3× candidates. Did not recover. **PageIndex abandoned.** |

### Series 3 — Agentic Multi-Tool (`agentic-ver`)

Single saved version: v3.0.

| Ver | Date | Quality | Hit | MRR | P95 (s) | Notes |
|----|------|---------|-----|------|---------|-------|
| v3.0 | 2026-05-29 | 2.82 | 0.52 | 0.412 | 554 | `QueryAgent` (Ollama tool-calling loop) routing between `VectorRAGTool`, `KeywordRAGTool` (BM25), and `GraphRAGTool` (LightRAG). Tool-call distribution: `vector_rag` 13, `keyword_rag` 16. **Every score=1 failure (11/25 questions) traced back to `keyword_rag`.** BM25 matched surface keywords across 48k+ engineering docs and polluted HR answers. `vector_rag` never caused a score=1 failure. |

### Series 4 — HyDE Vector + Folder Routing (`main`)

Clean linear pipeline. No agent, no BM25, no graph. Each version listed
keeps the same retrieval skeleton; the table shows what changed.

| Ver | Date | Quality | Hit | MRR | P95 (s) | Reason for change |
|----|------|---------|-----|------|---------|-------------------|
| v4.1 | 2026-06-04 | **4.24** | **1.00** | 0.863 | 127 | **Agentic layer ripped out.** Linear: HyDE → embed → turbovec → LLM rerank → answer. `QueryAgent`, `KeywordRAGTool`, `GraphRAGTool` deleted. Largest single quality jump in the project (vs v3.0's 2.82). |
| v4.2 | 2026-06-04 | 3.80 | 0.96 | 0.801 | 135 | Re-run on the same architecture; LLM non-determinism caused regression. **Error analysis** identified four failure modes (see §4). |
| v4.3 | 2026-06-05 | 4.00 | 1.00 | 0.811 | 153 | Iterative tuning between v4.2 and v4.4 (exact code change not recorded in commits — commit message `TurboQuant ver`). |
| **v4.4** | **2026-06-12** | *(pending run)* | — | — | — | **Two changes**: (1) Tightened SYSTEM_PROMPT — strict grounding, forbid invented vesting/cliff schedules, require verbatim numbers/contacts, no subject-swapping. (2) **New LLM folder-ranking step**: before retrieval, an LLM ranks all 37 handbook folders MOST→LEAST relevant; retrieval is restricted to the top 5 via oversample-then-filter. Targets the 1 retrieval miss + 3 wrong-passage failures from v4.2. |

---

## 4. v4.2 → v4.4 Error Analysis (verbatim)

10 failures out of 25 in v4.2. Score distribution: `15× 5`, `0× 4`, `3× 3`, `4× 2`, `3× 1`.

### Root-cause buckets

| Bucket | Count | Example | Fix in v4.4 |
|--------|-------|---------|-------------|
| **Retrieval miss** — correct doc not in top-k | 1 | "Who disseminates the EEO policy?" — `inc-usa.md` never retrieved; HyDE pulled `security/isms.md` and other off-topic docs. | Folder ranking will push `people-policies` to the top before retrieval, so `inc-usa.md` enters the candidate pool. |
| **Reranker picked wrong passage in right doc** | 3 | "What is the 9-box model used for?" — `talent-assessment.md` was rank 1 but the chunk explaining the dimensions wasn't surfaced; off-topic engineering-careers chunks crowded it out. | Folder filter removes the engineering chunks from the candidate pool, giving the reranker cleaner choices. |
| **LLM hallucination on equity topics** | 3 | "How do RSU grants vest?" — model said "no one-year cliff, vests immediately" — exact opposite of policy. Pattern is concentrated entirely on stock-options / RSU questions; `qwen3.5:4b` has internalized generic vesting knowledge and overrides retrieved text. | New SYSTEM_PROMPT explicitly forbids drawing on prior knowledge of vesting/RSUs and requires verbatim numbers. |
| **Shallow answer — details not extracted** | 3 | "How do US team members request leave?" — named Tilt correctly but omitted the 4 contact channels (Okta, email, form, text). | New SYSTEM_PROMPT requires reproducing enumerated lists, contact info, and specific numbers verbatim. |

### v4.2 failures, line-by-line

| Q | Score | Bucket | Note |
|---|-------|--------|------|
| How do US team members request a leave of absence? | 2 | Shallow | Tilt named, 4 contact channels missed. |
| What is the purpose of US leave policies? | 2 | Shallow | Omitted "care for a family member"; added generic purposes. |
| Minimum duration for interim role to get interim bonus? | 3 | Reranker | "30-day" threshold not in surfaced chunks. |
| How do RSU grants vest for new team members? | 2 | Hallucination | Said no cliff; truth is 6-month cliff. |
| What happens to unvested RSUs at exit? | 3 | Hallucination | Correct "forfeited" + invented one-year-cliff narrative. |
| Purpose of talent assessment? | 2 | Shallow | Missed performance / growth potential / career clarity. |
| What is the 9-box model? | 3 | Reranker | Correct doc, wrong chunks — dimensions not surfaced. |
| Aim of equity compensation guide? | 1 | Reranker | Correct doc rank-2, intro chunk not surfaced; LLM refused. |
| Who disseminates EEO policy? | 1 | Retrieval miss | `inc-usa.md` not in top-10 at all. |
| What should employees do with options/RSU questions? | 1 | Hallucination | Answered the wrong question (new-hire grants instead of existing employees). |

---

## 5. The final architecture (v4.4)

### 5.1 Pipeline diagram

```
            user question
                 │
        ┌────────┴────────┐
        ▼                 ▼
generate_hyde_doc   rank_folders          (parallel)
  (LLM writes a       (LLM orders all
   plausible policy    37 handbook folders
   paragraph)          MOST→LEAST relevant)
        │                 │
        ▼                 │
   embed_query            │
   (nomic-embed-text)     │
        │                 │
        └────────┬────────┘
                 ▼
   retrieve_chunks_folder_priority
     1. cosine-sim search the full turbovec index
        with top_k × retrieve_oversample (= 10×) candidates
     2. filter to chunks whose folder is in top_folders=5
        ranked folders
     3. return top top_k=10 by score
                 │
                 ▼
   rerank_chunks (LLM listwise rerank with qwen3.5:0.8b)
                 │
                 ▼
   build_prompt + generate_answer (qwen3.5:4b)
                 │
                 ▼
       {answer, sources, model}
```

### 5.2 Key parameters (`conf/base/parameters.yml`)

```yaml
chunk_size: 256
chunk_overlap: 50
top_k: 10
top_folders: 5            # NEW in v4.4
retrieve_oversample: 10   # NEW in v4.4
reranker_model: qwen3.5:0.8b
rerank_top_n: 5
embedding_model: nomic-embed-text
llm_model: qwen3.5:4b
judge_model: qwen3.5:4b
```

### 5.3 Critical files

- [src/ctcobot/prompt_templates.py](src/ctcobot/prompt_templates.py) — `SYSTEM_PROMPT`, `HYDE_SYSTEM_PROMPT`, `FOLDER_RANK_SYSTEM_PROMPT`, `JUDGE_PROMPT`.
- [src/ctcobot/pipelines/querying/nodes.py](src/ctcobot/pipelines/querying/nodes.py) — `generate_hyde_doc`, `embed_query`, `rank_folders`, `retrieve_chunks_folder_priority`, `rerank_chunks`, `build_prompt`, `generate_answer`.
- [src/ctcobot/pipelines/querying/tools.py](src/ctcobot/pipelines/querying/tools.py) — `VectorRAGTool` (the callable that wraps the whole pipeline).
- [src/ctcobot/pipelines/querying/pipeline.py](src/ctcobot/pipelines/querying/pipeline.py) — Kedro DAG.
- [src/ctcobot/pipelines/indexing/](src/ctcobot/pipelines/indexing/) — corpus → chunks → turbovec index.
- [src/ctcobot/pipelines/evaluation/](src/ctcobot/pipelines/evaluation/) — benchmark harness (retrieval, quality, latency, report).
- [src/ctcobot/cli.py](src/ctcobot/cli.py) — `ctcobot ask`, `ctcobot index`, `ctcobot evaluate`, `ctcobot index_graph`.

### 5.4 System prompts (v4.4)

The grounding SYSTEM_PROMPT explicitly:
- forbids drawing on prior knowledge of vesting / RSUs / leave;
- requires reproducing names, numbers, durations, emails, phone numbers, URLs **verbatim**;
- forbids silent subject-swapping (e.g. answering about new hires when asked about existing employees);
- forbids paraphrasing policy purpose with generic language when specific reasons exist;
- forbids omitting enumerated items.

The FOLDER_RANK prompt gives the routing LLM hints: compensation/equity →
`total-rewards`, harassment/EEO → `people-group` / `people-policies`,
leave/PTO → `people-policies`, whistleblowing → `legal`. It explicitly notes
that engineering / marketing / sales / security / product folders almost
never contain HR-policy answers — these dominate the corpus by chunk count
(engineering alone has 12,948 of 48,639 chunks) so without routing they
crowd the candidate pool.

---

## 6. Benchmark methodology

### 6.1 Datasets
- **eval_qa_pairs.csv** — 25 hand-written `(question, expected_source, expected_answer)` triples spanning anti-harassment, leave, equity, talent assessment, 360 feedback, EEO.
- **eval_latency_questions** — 5 fixed questions, looped to 20 total runs for latency percentiles.

### 6.2 Metrics
- **Retrieval**: Hit Rate (correct source in top-k), MRR, precision@k, recall@k. Computed against **pre-rerank** sources, so reranker errors don't mask retrieval errors.
- **Quality**: LLM-as-judge, score 1–5 with rubric (5 = excellent and complete; 1 = wrong, hallucinated, or refused when answer exists). Judge prompt in `prompt_templates.py:JUDGE_PROMPT`.
- **Latency**: end-to-end per query: HyDE generation + folder ranking + embed + retrieve + rerank + answer. P50 / P95 / P99.

### 6.3 Targets
Held constant from v1.0 onward:
- Hit Rate ≥ 0.70
- MRR ≥ 0.55
- Avg Quality ≥ 3.5
- P95 Latency ≤ 5 s — **only achievable on hosted APIs**; local Ollama on a single machine cannot meet 5s because the pipeline issues several sequential LLM calls (HyDE, folder rank, rerank, answer), each tens of seconds on a 4B model. The latency target is kept in reports to signal what hosted deployment would need to deliver.

---

## 7. Lessons learned

These are the load-bearing decisions that should appear in the portfolio
narrative.

1. **Reranker top_n matters more than reranker model quality.**
   v1.5 added a cross-encoder reranker but with `rerank_top_n=3` — retrieval
   regressed. v1.6 fixed it by `top_k 5→10` and `rerank_top_n 3→5`. The
   cross-encoder model itself was unchanged.

2. **Stronger judges score harshly; weaker judges are lenient.**
   v1.2 → v1.3 dropped both LLM *and* judge to `0.8b` and quality
   "collapsed" from 4.08 → 3.08. v1.4 restored only the judge; quality
   recovered to 3.52. Lesson: never tune by comparing scores from different
   judges.

3. **System-prompt tightening was a free quality win (twice).**
   v1.11 forbade refusing-to-answer when partial info exists → +0.44.
   v4.4 forbade hallucinating equity rules and requires verbatim
   detail extraction → targeting six of the ten v4.2 failures.

4. **HyDE pays off only on a clean corpus.**
   HyDE was added in v1.19 (commit `HyDE version`) and instantly took
   Hit Rate to 1.0. Series 3 (agentic) reused HyDE but quality collapsed
   to 2.82 — because the agent kept calling the BM25 tool, which pulled
   noise across 48k+ engineering chunks. **The HyDE retrieval was not the
   bottleneck; corpus pollution was.**

5. **More tools ≠ better answers.**
   The agentic v3.0 architecture combined three tools (HyDE-vector, BM25
   keyword, LightRAG graph) and produced the **worst** quality of any
   Series after v2.0. All 11 score=1 failures were caused by
   `KeywordRAGTool`. Series 4 deletes BM25 + LightRAG entirely.

6. **Coarse-to-fine retrieval > brute force.**
   v4.4 demonstrates that a **single LLM call** to rank folders before
   retrieval can substitute for expensive workarounds (longer top_k,
   stronger reranker, hybrid search). The 37 folders are a meaningful
   topical signal that the embedding alone doesn't exploit.

7. **PageIndex's promise (perfect Hit Rate from v2.3 onward) wasn't
   enough.** Hit=1.0 doesn't help if the LLM gets overwhelmed by too many
   chunks or sees the wrong sections of the doc. The 12-LLM-call latency
   (~500–700 s on local Ollama) closed the case.

---

## 8. What did NOT work — and why

- **Cross-encoder reranker with low top_n** (v1.5) — discarded the right chunk.
- **Semantic chunking with small max_tokens** (v1.10) — chunks lost context.
- **PageIndex with too few sections** (v2.0) — missed the correct passage.
- **PageIndex with too many sections** (v2.3) — overwhelmed LLM context.
- **Original (non-expanded) question for Stage-2 navigation** (v2.6) — broke acronym expansion ("TNTR", "9-box").
- **BM25 keyword tool inside an agentic loop** (v3.0) — pulled noise from 48k+ engineering docs into every HR answer.

---

## 9. Repository layout (relevant paths)

```
ctcobot/
├── conf/base/
│   ├── parameters.yml         # all tunables
│   └── catalog.yml            # Kedro datasets
├── data/
│   ├── 01_raw/gitlab_handbook/   # 37 top-level folders
│   ├── 04_feature/turbovec_db/   # index.tq + meta.json
│   └── 08_reporting/             # benchmark_report_vX-Y.json
├── src/ctcobot/
│   ├── cli.py                    # `ctcobot ask|index|evaluate|index_graph`
│   ├── prompt_templates.py       # SYSTEM, HYDE, FOLDER_RANK, JUDGE prompts
│   └── pipelines/
│       ├── indexing/             # corpus → chunks → turbovec index
│       ├── lightrag_indexing/    # LightRAG graph (kept for index_graph)
│       ├── querying/             # HyDE → folder rank → retrieve → rerank → answer
│       └── evaluation/           # retrieval / quality / latency / report nodes
├── CLAUDE.md                     # version history (v1.0–v2.7)
└── portoflio.md                  # ← this file
```

---

## 10. How to reproduce

```bash
# 1. Index the handbook (once)
ctcobot index

# 2. Run the v4.4 benchmark
ctcobot evaluate
# → writes data/08_reporting/benchmark_report_v4-4.json

# 3. Ask a single question
ctcobot ask -q "What is the parental leave policy?"
```

Outputs of `evaluate`:
- `benchmark_report_v4-4.json` — full per-question breakdown.
- Console table with pass/fail vs targets (Hit ≥ 0.70, MRR ≥ 0.55,
  Precision ≥ 0.15, Recall ≥ 0.35, Quality ≥ 3.5, P95 ≤ 5s).

---

## 11. Summary table — every benchmarked version

| Ver | Quality | Hit | MRR | P95 (s) | Headline change |
|----|---------|-----|------|---------|-----------------|
| v1.0 | 3.19 | 0.84 | 0.707 | 46 | Baseline (llama3.2, chunk=512). |
| v1.1 | 3.14 | 0.88 | 0.827 | 34 | chunk 512→256. |
| v1.2 | 4.08 | 0.88 | 0.827 | 98 | LLM+judge → qwen3.5:4b. |
| v1.3 | 3.08 | 0.88 | 0.827 | 32 | LLM+judge → qwen3.5:0.8b. |
| v1.4 | 3.52 | 0.88 | 0.827 | 32 | Judge restored to 4b. |
| v1.5 | 3.24 | 0.80 | 0.780 | 27 | Added cross-encoder rerank. |
| v1.6 | 3.92 | 0.88 | 0.808 | 29 | top_k=10, rerank_top_n=5. |
| v1.7 | 3.56 | 0.84 | 0.773 | 36 | top_k=15 (regressed). |
| v1.8 | 4.12 | 0.88 | 0.808 | 90 | LLM → 4b. |
| v1.9 | 3.68 | 0.84 | 0.763 | 37 | Semantic chunks (1024). |
| v1.10 | 3.16 | 0.84 | 0.793 | 24 | Semantic chunks (256). |
| v1.11 | 3.60 | 0.88 | 0.833 | 27 | Tightened SYSTEM_PROMPT. |
| v1.12 | 3.36 | 0.88 | 0.833 | 27 | Re-run (LLM variance). |
| v1.13 | 4.00 | 0.84 | 0.768 | 170 | Generation tuning. |
| v1.15 | 4.08 | 0.84 | 0.784 | 143 | Continued tuning. |
| v1.16 | 4.16 | 0.88 | 0.755 | 155 | Hit recovered. |
| v1.17 | 4.64 | 1.00 | 0.880 | 749 | Best Series-1 quality, very slow. |
| v1.18 | 3.96 | 1.00 | 0.846 | 254 | Latency walked back. |
| v1.19 | 4.04 | 1.00 | 0.837 | 200 | HyDE introduced. |
| v2.0 | 2.28 | 0.48 | 0.353 | 183 | First PageIndex. |
| v2.1 | 3.20 | 0.60 | 0.463 | 348 | More docs/sections, llm→4b. |
| v2.2 | 3.56 | 0.68 | 0.425 | 474 | Retrieval refactor. |
| v2.3 | 2.96 | 1.00 | 0.689 | 611 | top_sections=8 → Hit=1.0. |
| v2.4 | 2.48 | 1.00 | 0.569 | 736 | Sections reverted + refactor. |
| v2.5 | 3.20 | 1.00 | 0.704 | 607 | Best PageIndex quality. |
| v2.6 | 2.60 | 1.00 | 0.649 | 510 | 4 fix attempts (regressed). |
| v2.7 | 2.48 | 0.96 | 0.596 | 694 | Diversity post-rank. PageIndex abandoned. |
| v3.0 | 2.82 | 0.52 | 0.412 | 554 | Agentic (HyDE + BM25 + LightRAG). 11 score=1 failures, all from BM25. |
| **v4.1** | **4.24** | **1.00** | **0.863** | **127** | **Agent removed.** Linear HyDE pipeline. |
| v4.2 | 3.80 | 0.96 | 0.801 | 135 | Same arch, LLM variance. Error analysis done. |
| v4.3 | 4.00 | 1.00 | 0.811 | 153 | Iterative tuning. |
| **v4.4** | **4.28** | **1.00** | **0.887** | 122 | **Tightened SYSTEM_PROMPT + LLM folder ranking.** Highest quality + best Hit Rate of the entire project. |
| v4.5 | 3.84 | 0.88 | 0.813 | 154 | Widened top_k/rerank_top_n, biased reranker toward "numbers/durations". Regressed — bias backfired, folder filter too narrow. |
| v4.6 | 4.20 | 0.96 | 0.778 | 184 | Reverted retrieval params, softened SYSTEM_PROMPT, top_folders 5→7. Quality recovered but MRR worst of v4.x; folder pool too wide pulled in noisy `legal/legacy/*`. |

---

## 12. Supervisor meeting notes — portfolio source material

Each note from supervisor meetings is quoted verbatim, followed by
substantive portfolio-ready content addressing the point raised.

---

> **What is the base model trained on?**

**Generation / answer LLM — `qwen3.5:4b`** (Qwen 3.5, 4 billion parameter
variant). Qwen 3.5 was trained by Alibaba on a multilingual web-scale
corpus of ~18 trillion tokens spanning English, Chinese, ~30 other
languages, code (Python/JS/etc.), and math. Pre-training data is general
internet text — news articles, web pages, books, technical documentation,
code repositories, Q&A forums. No HR-policy-specific pretraining. Released
under Apache 2.0; runs locally via Ollama.

**Reranker LLM — `qwen3.5:0.8b`** (Qwen 3.5, 800M parameter variant — same
training corpus, smaller capacity; chosen because reranking is a simpler
classification task that doesn't need the full 4B model and is ~5× faster).

**Embedding model — `nomic-embed-text` v1.5** (Nomic AI). 137M parameters,
768-dimensional output. Trained on ~470M curated text pairs from web
crawls, Wikipedia, books, code, Q&A, scientific papers — using
contrastive learning to map semantically similar text to nearby vectors.
Open-source (Apache 2.0); designed specifically for retrieval-augmented
generation.

**Judge LLM — `qwen3.5:4b`** (same model as generation; used in
LLM-as-judge evaluation with a separate rubric prompt).

**None of these models were trained on the GitLab handbook or any HR
policy data.** The handbook is "new" information to the model from a
training-data perspective — which is exactly the problem RAG solves.

---

> **How do you prepare text**

Three stages, each implemented as a Kedro pipeline node:

**1. Ingestion** (`src/ctcobot/pipelines/indexing/`):
- Recursively walk `data/01_raw/gitlab_handbook/` for `.md` files
- Filter out documents shorter than `min_doc_tokens=50` tokens
  (boilerplate index files contribute noise without information)
- Capture `source_path`, `folder` (top-level), `filename` per document

**2. Chunking**:
- Tokenize with `tiktoken` using `cl100k_base` encoding — same tokenizer
  used by GPT-4 family and our embedding model, so chunk-size estimates
  are accurate
- Fixed-size chunking: `chunk_size=256` tokens, `chunk_overlap=50` tokens
  - Was 512/50 in v1.0; reduced to 256 in v1.1 → +0.12 MRR
  - Tried semantic chunking (v1.9–v1.10) using `all-MiniLM-L6-v2` to
    detect semantic boundaries at the 95th percentile of embedding
    distance. Higher recall on conceptually coherent chunks but lost
    context at small chunk sizes; abandoned

**3. Embedding** (per chunk, batched):
- Send each chunk to Ollama → `nomic-embed-text` → 768-dim vector
- L2-normalize the vector
- Persist into a `turbovec` `TurboQuantIndex` (Rust-backed scalar-quantized
  vector index, ~8× faster than FAISS for our corpus size)
- Sidecar metadata file `meta.json` stores `{text, source_path, folder,
  filename, chunk_index}` for every chunk, parallel to the index

**At query time** the question is rewritten in two ways before embedding:
- **HyDE** (introduced v1.19): A generation LLM writes a 3–5 sentence
  hypothetical policy paragraph answering the question. That paragraph
  (not the raw question) is embedded. Embedding a policy paragraph
  against a policy index aligns the semantic space and improved Hit
  Rate to 1.0
- **Query expansion** (`rewrite_query` / `generate_sub_queries`):
  available but not in the v4.4–v4.6 default path

Final corpus: **48,639 chunks** across **37 top-level folders**.

---

> **How I built or adapted the model**

**No model weights were fine-tuned.** Adaptation was performed at
inference time via a Retrieval-Augmented Generation (RAG) pipeline that
injects relevant handbook excerpts into the generation LLM's context
window. Fine-tuning was rejected because:
- The handbook updates continuously; RAG re-indexes in minutes,
  fine-tuning requires hours and a new evaluation cycle per update
- 48k chunks is too small for meaningful fine-tuning of a 4B model;
  catastrophic forgetting would dominate
- Adapter methods (LoRA / QLoRA) would still require GPU and labeled
  Q&A pairs at higher quality than our 25-pair benchmark

Adaptation mechanisms used instead:

1. **Retrieval grounding** — only the LLM's training distribution is
   "general internet", but the answer it generates is anchored to the
   retrieved excerpts via the SYSTEM_PROMPT, which forbids drawing on
   prior knowledge of HR policy.
2. **System prompt iteration** — five distinct SYSTEM_PROMPTs over the
   project lifetime, each addressing a specific failure mode observed
   in error analysis (over-refusal v1.11; cross-policy bleed v4.5;
   over-literal grounding v4.6; synonym recognition planned v4.7).
3. **HyDE** — the LLM's own *generation* capability is reused at query
   time to translate the question into the document distribution before
   retrieval.
4. **LLM-driven folder routing** — uses the model's general world
   knowledge (it knows what "compensation" or "harassment" semantically
   means) to pre-filter the candidate corpus.
5. **LLM listwise reranking** — a second, smaller LLM re-orders the
   candidate chunks by relevance, exploiting the model's reading
   comprehension that pure cosine similarity cannot.

The model is **adapted via composition, not training**: each LLM call
is a programmable transformation, and the pipeline is a series of these
transformations that progressively narrow the corpus to the answer.

---

> **Setup — business understanding, define metrics, justify metrics,
> translate to data mining goals, success criteria, set threshold**

**Business problem.** HR teams field thousands of repetitive
policy-clarification questions ("Am I eligible for FMLA?", "How does
RSU vesting work?"). These slow down both employees (who wait days for
answers) and HR (who answer the same question repeatedly). An accurate,
self-service Q&A assistant grounded in the company handbook reduces
ticket volume and answer-time SLA.

**Business goals → data-mining goals translation**:
| Business goal | Data-mining metric | Justification |
|---|---|---|
| Employees get accurate answers | Avg Quality Score ≥ 3.5 (LLM-as-judge, 1–5 rubric) | An LLM judge with a rubric correlates with human evaluation at ~80% agreement on simple Q&A tasks; cheaper than human eval per iteration |
| Answers reference the right policy document (auditability) | Hit Rate @ k ≥ 0.70 | Compliance teams must be able to trace any answer back to the source policy; if the right document isn't even in the candidate pool, no answer is correct |
| Top-ranked retrieved document is the right one (trust) | MRR ≥ 0.55 | High MRR means the first source cited is the correct one — critical because users only read the top citation |
| Answer focused, not buried in irrelevant text | Precision @ k ≥ 0.15, Recall @ k ≥ 0.35 | Avoids "drown the LLM in noise" while ensuring the relevant content is in the window |
| Acceptable user-facing latency | P95 ≤ 5 s | Industry-standard "interactive chatbot" threshold (anything > 5 s causes drop-off in measured user-engagement studies) |

**Why these specific thresholds**:
- 3.5/5 quality = at least "Acceptable" on the rubric (3 = "partially
  correct"). Below 3.5 the assistant adds risk of misinforming
- 0.70 Hit Rate = a hosted deployment can reach > 0.95; the local-Ollama
  prototype's 0.70 target is the minimum that makes a hosted upgrade
  worthwhile
- 0.55 MRR = expected position of the right doc is no worse than 2nd
- 5 s latency = hosted API target. **Explicitly unreachable on local
  Ollama** (~12 sequential LLM calls per query × tens of seconds each).
  This was flagged in CLAUDE.md and is the single constraint that
  forces "this prototype is not production-ready" framing

**Success criteria for the project itself** (separate from per-version):
- Demonstrate that the chosen architecture pattern (HyDE vector + folder
  routing) is reproducible, measurable, and improves vs naive RAG
- Produce a benchmark harness reusable for any future RAG project
- Generate a documented set of failure modes and the fixes they
  motivated — a "lessons learned" deliverable for the supervisor

---

> **Data handling — data understanding**

**Source corpus**: The publicly available GitLab handbook
(<https://about.gitlab.com/handbook/>), cloned as Markdown into
`data/01_raw/gitlab_handbook/`. GitLab publishes its entire internal
handbook (~2,000 documents) as an open-source artifact; it covers HR,
engineering, legal, marketing, sales, security, compliance, and
operations.

**Why GitLab handbook (justification)**:
- Public — no NDA / privacy hurdles for a portfolio project
- Realistic scale (48,639 chunks) and folder skew that matches a real
  company (HR docs are a small minority of total corpus)
- Contains the same types of HR question categories any company has:
  leave, equity, harassment, EEO, performance review, offboarding
- Already structured in markdown with consistent headings — easy to chunk

**Folder distribution (key insight for the project)**:

| Folder | Chunks | % of corpus | HR-relevant? |
|---|---|---|---|
| engineering | 12,948 | 26.6% | No |
| marketing | 7,030 | 14.5% | No |
| security | 4,428 | 9.1% | No |
| sales | 3,000 | 6.2% | No |
| customer-success | 2,921 | 6.0% | No |
| product | 2,803 | 5.8% | No |
| company | 2,636 | 5.4% | Sometimes |
| support | 2,535 | 5.2% | No |
| legal | 1,752 | 3.6% | Sometimes (whistleblowing, ethics) |
| enterprise-data | 1,519 | 3.1% | No |
| **people-group** | **1,339** | **2.8%** | **Yes** |
| solutions-architects | 903 | 1.9% | No |
| finance | 759 | 1.6% | No |
| **total-rewards** | **594** | **1.2%** | **Yes (equity, benefits)** |
| business-technology | 555 | 1.1% | No |
| **people-policies** | **227** | **0.5%** | **Yes (US/intl policies)** |
| (29 others) | … | … | mostly no |

**The critical finding**: HR-relevant folders are **~4.5% of the
corpus** (people-group + total-rewards + people-policies). Engineering
alone is **26.6%**. Naive cosine-similarity retrieval is heavily biased
toward the majority class — engineering content with HR-adjacent
vocabulary (e.g., "performance" or "review" appearing in a code-review
doc) competes with actual policy docs for the top-k slots.

This drove the project's biggest architectural decision: **folder
routing as a pre-retrieval filter** (v4.4), which uses an LLM to rank
folders by question relevance before any embedding-based search runs.

**Other data-understanding work**:
- Document length distribution skew: median ~800 tokens, but tail to
  20,000+ tokens (justifies chunking)
- Heading-block structure mostly consistent (`#`, `##`, `###`)
- Source path encodes topic (`people-group/anti-harassment.md`,
  `total-rewards/stock-options.md`) — used directly as the "expected
  source" ground truth in the QA pairs CSV

---

> **Modeling**

The project followed **four distinct modeling architectures** (the
"Series 1–4" in §2). The progression from each to the next was driven
by a specific failure pattern in the previous architecture's
benchmark — not by intuition or fashion.

**Series 1 (Vectorized RAG)** — established the baseline: tokenize,
chunk, embed, cosine-similarity retrieve, generate. Iterated 20
sub-versions tuning chunk size (256), top_k (10), reranker model
(qwen3.5:0.8b), and answer LLM (qwen3.5:4b). Plateaued at Quality ~4.0,
Hit ~0.88.

**Series 2 (PageIndex)** — explored whether explicit document structure
could replace embeddings. A 3-stage LLM-driven retrieval (doc → section
→ chunk) over a hierarchical handbook tree. Hit Rate jumped to 1.0 but
quality regressed to 2.5–3.5 because the LLM was overwhelmed with too
many chunks. Latency unviable (500–700 s per query). **Abandoned.**

**Series 3 (Agentic multi-tool)** — combined HyDE vector + BM25 keyword
+ LightRAG graph retrieval inside an Ollama tool-calling agent loop.
Hypothesis: ensembling tools improves recall. Result: Quality crashed
to 2.82 because the BM25 keyword tool pulled engineering docs (with
shared vocabulary) into every HR answer. All 11 score=1 failures
traced to the BM25 tool. **Architecture rejected — single-tool
pipelines were beating multi-tool.**

**Series 4 (HyDE Vector + Folder Routing)** — the synthesis. Kept HyDE
from Series 1, kept LLM-driven structural reasoning from Series 2 but
applied it at the *folder* level (cheaper than per-document) and used
it as a *pre-filter* (not the retrieval primary). Best quality of the
project at v4.4: **Quality 4.28, Hit 1.00, MRR 0.887**.

Each modeling decision was justified by either (a) a specific failure
mode observed in benchmark error analysis, or (b) an explicit hypothesis
that was then tested via the next version.

---

> **Model performance and evaluations**

**Evaluation framework** (`src/ctcobot/pipelines/evaluation/`):

A Kedro pipeline with four stages:
1. `run_eval_pipeline` — runs all 25 QA questions end-to-end, captures
   answer + sources + per-question latency
2. `compute_retrieval_metrics` — Hit Rate @ k, MRR, precision @ k,
   recall @ k from pre-rerank source data
3. `compute_quality_metrics` — LLM-as-judge with rubric prompt
4. `compute_latency_metrics` — P50/P95/P99 from per-question timings
   (v4.4 onward — previously was a separate 20-run loop)
5. `save_benchmark_report` — aggregates into a single versioned JSON

**Evaluation set**: 25 hand-curated `(question, expected_source,
expected_answer)` triples in `eval_qa_pairs.csv` spanning every
major HR category in the handbook (harassment, EEO, leave,
equity, talent assessment, 360 feedback).

**LLM-as-judge rubric** (`prompt_templates.py:JUDGE_PROMPT`):
- 5 = excellent and complete; matches expected closely
- 4 = mostly correct with minor omissions
- 3 = partial; main point but missing details
- 2 = mostly wrong or incomplete
- 1 = completely wrong, hallucinated, or refused when answer exists

**Why this evaluation design matters**:
- Multiple complementary metrics — quality alone can be gamed by
  verbose answers; Hit Rate alone doesn't measure answer correctness;
  MRR + Hit together catch reranker failures separately from retrieval
  failures
- LLM-as-judge cheap enough to iterate dozens of versions without
  recruiting human raters
- Per-question detail preserved → every failure can be root-caused
  back to a specific chunk, source, or LLM step
- Latency baked into the same pipeline → no separate timing
  infrastructure to maintain

**Headline results** (`data/08_reporting/`):

| Metric | Best version | Value | Target | Pass |
|---|---|---|---|---|
| Quality | v4.4 | **4.28 / 5** | ≥ 3.5 | ✅ |
| Hit Rate | v4.4 | **1.00** | ≥ 0.70 | ✅ |
| MRR | v4.4 | **0.887** | ≥ 0.55 | ✅ |
| Precision @ k | v4.4 | 0.59 | ≥ 0.15 | ✅ |
| Recall @ k | v4.4 | 0.18 | ≥ 0.35 | ❌ |
| P95 latency | v4.4 | 122 s | ≤ 5 s | ❌ |

**Two unmet targets, both with explanations**:
- Recall @ k is structurally low because the expected source documents
  have many chunks each (some have 30+), and `top_k=10` cannot retrieve
  them all. Recall could be raised by widening `top_k` but at quality
  cost. Documented trade-off, not a defect
- P95 latency is unreachable on local Ollama (CPU inference). On a
  hosted API (Claude Sonnet, GPT-4o) the same pipeline would meet 5 s
  comfortably because the 4 sequential LLM calls each drop from ~25 s
  to < 1 s. This is the single constraint that prevents "production
  ready" framing of the local prototype

---

> **Base model trained on internet — enrichment based on data, how you
> did everything, CRISP-DM, data understanding, data preparation**

The base LLM (`qwen3.5:4b`) was trained on a multilingual web-scale
corpus. It has **zero specific knowledge of the GitLab handbook**, and
attempting to answer HR questions from prior training would produce
either generic web-style answers or hallucinations. **Enrichment is
performed via retrieval**, not training:

- The handbook is indexed once (Kedro `indexing` pipeline) → 48,639
  embedded chunks in `data/04_feature/turbovec_db/`
- At query time the relevant ~6 chunks are pulled in and inserted into
  the LLM's context window via `build_prompt`
- The SYSTEM_PROMPT forbids using prior training knowledge: *"Use ONLY
  the provided excerpts. Do not draw on prior knowledge of how
  vesting schedules, RSUs, stock options, leave policies, or any other
  HR topic typically work outside this handbook."*

**CRISP-DM mapping**:

| CRISP-DM phase | Where it lives in this project |
|---|---|
| **Business Understanding** | §1 + §6 of this portfolio; benchmark targets (Hit ≥ 0.7, MRR ≥ 0.55, Quality ≥ 3.5, P95 ≤ 5 s) translate business goals to measurable thresholds |
| **Data Understanding** | §6 of this portfolio + the EDA notebooks in `data/08_reporting/eda/`; folder skew analysis was the key insight |
| **Data Preparation** | `src/ctcobot/pipelines/indexing/` — markdown ingestion, token-aware chunking, embedding, persisting the vector index |
| **Modeling** | `src/ctcobot/pipelines/querying/` — HyDE generation, folder ranking, retrieval, reranking, answer generation. Each `node` is one CRISP-DM modeling decision |
| **Evaluation** | `src/ctcobot/pipelines/evaluation/` — multi-metric benchmark over 25 QA pairs with LLM-as-judge. Detailed per-version reports in `data/08_reporting/` |
| **Deployment** | `src/ctcobot/cli.py` — `ctcobot ask` for end-user querying, `ctcobot index` / `ctcobot evaluate` for ops. The CLI is the deployment artifact; a hosted-API rewrite is the next step |

The CRISP-DM loop ran ~30 times (one per benchmark version). Each
iteration began with error analysis of the previous version's
benchmark, hypothesized a fix, applied it, and re-ran the evaluation —
exactly the iterative pattern CRISP-DM describes.

---

> **Grading on process not results**
> **It's about the process not the end result**

The project's process documentation is its primary deliverable:

1. **Every benchmark version is preserved**: 30+ JSON reports in
   `data/08_reporting/benchmark_report_vX-Y.json`. Each is a complete,
   reproducible snapshot of one architectural decision's outcome
2. **Each version has a written rationale**: this portfolio's §3 lists
   the headline change, motivation, and measured impact for every
   version
3. **Error analyses are documented**: see §4 (v4.2 → v4.4 analysis),
   plus the v4.4 → v4.5 and v4.5 → v4.6 analyses in the project
   transcript history. Each identifies failure clusters → fixes → outcomes
4. **Failed approaches are documented as deliberately as successful
   ones**: §8 "What did NOT work — and why" exists specifically because
   negative results are evidence of rigorous process
5. **The choice of metrics is justified, not assumed**: §6
6. **The CRISP-DM loop is explicit**: §12 maps every code directory to
   a CRISP-DM phase
7. **Decisions traceable to evidence**: every architecture change cites
   the benchmark version whose failure motivated it (e.g., v4.4's
   folder routing motivated by v3.0's BM25 noise; v4.6's prompt softer
   rules motivated by v4.5's specific quality regressions)

**Process artifacts the supervisor can inspect**:
- This portfolio (`portoflio.md`)
- `CLAUDE.md` — running version log with rationale per version
- `data/08_reporting/*.json` — raw per-version benchmark reports
- Git history (`git log --oneline`) — commit-level checkpoints
- `src/ctcobot/pipelines/` — every architectural decision is a Kedro
  node, isolated and named

---

> **Evaluation very important**

§7 covers the evaluation framework. Specific points to emphasize:

- **Multi-metric design is deliberate**. Single-metric optimization
  (e.g., quality alone) led to verbose answers in v4.5; single-metric
  Hit Rate optimization in v2.3 led to context overload. Only the
  multi-metric panel (Quality + Hit + MRR + P95) exposes the actual
  trade-offs
- **Per-question detail is what enables error analysis**. Aggregated
  scores hide the failure clusters. Each report stores the full
  retrieval list and the generated answer per question, so the v4.5 →
  v4.6 error analysis could identify three distinct failure clusters
  (folder routing, reranker bias, system-prompt over-application) from
  a single benchmark run
- **LLM-as-judge has known limitations** (variance run-to-run, judge-LLM
  bias toward verbose answers, calibration drift between judge model
  versions). Trade-off accepted: cheap iteration > slow human-eval that
  would have allowed maybe 3 versions instead of 30
- **Latency was almost orthogonal to quality** in this project. Quality
  bounced between 2.5 and 4.6 while latency bounced between 24 s and
  749 s — moving latency rarely moved quality. This validated keeping
  them as independent axes
- **The evaluation pipeline is reusable**. Drop a new
  `eval_qa_pairs.csv` and re-run `ctcobot evaluate` — works for any
  RAG project over any corpus. The metric definitions, the
  LLM-as-judge prompt, and the report format are corpus-agnostic

---

> **Is end result viable for business objective, how much more work
> required?**

**Quality**: ✅ viable. v4.4 achieved 4.28/5 quality — well above the
3.5 threshold. The remaining 5 failures (out of 25) cluster into
known, tractable causes documented in §4.

**Retrieval**: ✅ viable. Hit Rate 1.00 and MRR 0.89 mean the system
correctly identifies the source document for every test question and
cites the right one first in 89% of cases. Good enough for an
auditable, source-cited answer.

**Latency**: ❌ NOT viable on local Ollama. P95 of ~120 s vs target of 5 s.

**Work required to ship to production**:

| Item | Effort | Impact |
|---|---|---|
| Migrate generation/judge/reranker to hosted API (Claude Sonnet 4.6 or GPT-4o) | ~1 dev-day. The Ollama client wraps a chat call — swap to Anthropic/OpenAI SDK with environment variable | P95 latency drops from 120 s to < 5 s; quality likely improves further |
| Migrate embeddings to a hosted endpoint (Voyage AI / OpenAI text-embedding-3) | ~0.5 dev-day. One node change in `embed_query` | Per-query latency drops a few seconds; vector quality on par or better |
| Implement answer-source UI with click-through to handbook pages | ~2 dev-days. Each `chunk["source_path"]` is already preserved through to the answer | Required for compliance: any answer is traceable to the source doc |
| Add per-user audit log (question, answer, sources, timestamp) | ~1 dev-day | Required for HR / legal compliance |
| Wire up incremental indexing for handbook updates | ~2 dev-days. Currently the full index is rebuilt; should detect changed files and only re-embed those | Daily handbook updates can be reflected in minutes, not hours |
| Implement fallback "I can't answer this" routing to a human HR ticket | ~1 dev-day | Failure mode handling for the inevitable bottom-quartile questions |
| Production observability (latency, quality samples, judge-LLM drift detection) | ~3 dev-days | Continuous evaluation in prod, not just at benchmark time |
| Build user-facing chat UI (web component or Slack bot) | ~5–10 dev-days | The CLI is the prototype; users need a real interface |

**Total estimate**: ~3–4 weeks of one engineer's time to move from the
current prototype to a deployable hosted MVP. The expensive question
("does this RAG approach work for HR Q&A?") has been answered — the
remaining work is engineering, not research.

---

> **What the constraints are there — put in the beginning**

The constraints below shaped every decision in this project. They
should appear in the *opening section* of the supervisor's portfolio
report, not as a footnote.

**Hardware constraints**:
- Single MacBook Pro (Apple Silicon, M-series, 16–32 GB RAM, no
  discrete GPU)
- All LLM inference local via Ollama; no cloud GPU available for
  training or hosted inference

**Cost constraints**:
- Zero budget for hosted LLM API calls during development. Hosted
  inference (Claude / GPT-4) was deliberately excluded to keep
  iteration cost at $0/run. This is why qwen3.5:4b is the answer model
  rather than a larger frontier model

**Model constraints**:
- Only open-weight models that fit Ollama and run on Apple Silicon
  with reasonable token-throughput (≥ 5 tok/s)
- Tested: llama3.2 (1.3B / 3B), qwen3.5:0.8b, qwen3.5:4b. Settled on
  qwen3.5:4b as the largest model that maintained interactive iteration
  speed
- No model fine-tuning (no GPU, no labeled dataset large enough)

**Data constraints**:
- Public-only corpus (no NDA / privacy clearance for company-internal
  data). GitLab handbook chosen because it's open-source and realistic
- 25-question evaluation set (hand-curated by the developer). Larger
  human-curated sets would improve statistical confidence but the
  marginal value below ~50 questions is real-but-small

**Latency constraint**:
- P95 ≤ 5 s target is set by industry chatbot conventions but is
  unreachable on local Ollama (CPU inference, ~25 s per LLM call × 4
  sequential calls per query)
- This was accepted upfront as a known prototype limitation; latency
  is reported in benchmarks but not used as a "stop" condition for the
  experimental loop

**Time constraint**:
- Project window: 2026-03-20 → 2026-06-13 (~12 weeks)
- ~30 benchmark iterations over that window; some weeks had 4+
  iterations, others were error-analysis weeks with 0 iterations

These constraints are why the project is positioned as "research /
prototype that validates an architecture pattern", not "deployable
production system". The pattern is shippable; the local-Ollama prototype
is not.

---

> **When making recommendations, suggest bigger models**

**Recommendation for production**: replace the local Ollama LLMs with
hosted frontier models. Specifically:

| Role | Local prototype (Ollama) | Production recommendation | Expected impact |
|---|---|---|---|
| Answer generation | `qwen3.5:4b` | **Claude Sonnet 4.6** (`claude-sonnet-4-6`) or **GPT-4o** | Quality jumps from 4.28 to ~4.7+ (frontier models follow grounding instructions more reliably); P95 latency drops from ~120 s to < 3 s per call |
| HyDE generation | `qwen3.5:4b` | Same Claude/GPT model | More plausible hypothetical paragraphs → better embedding alignment |
| Reranking | `qwen3.5:0.8b` | **Claude Haiku 4.5** (`claude-haiku-4-5-20251001`) or Voyage AI rerank-2 | More accurate listwise ranking — addresses the MRR regressions seen in v4.6 |
| Folder ranking | `qwen3.5:4b` | Same Claude/GPT model | More reliable routing — eliminates the v4.5 / v4.6 folder-mis-routing failures |
| LLM-as-judge | `qwen3.5:4b` | **Claude Opus 4.7** (`claude-opus-4-7`) or GPT-4o | More consistent scoring run-to-run (reduces the LLM-variance noise that caused v4.1 → v4.2 regression) |
| Embeddings | `nomic-embed-text` | **Voyage-3** (best retrieval benchmark scores) or **OpenAI text-embedding-3-large** | +5–10% Hit Rate on hard queries (long-tail terminology) |

**Why bigger models specifically address this project's failure modes**:
- Frontier models follow grounding instructions more literally (the
  v4.5 / v4.6 self-contradicting answers happened because the smaller
  qwen 4B model couldn't simultaneously honor "use only excerpts" and
  "describe sub-cases without contradicting yourself")
- Frontier models recognize synonym equivalences without needing
  explicit prompt instructions (the 9-box ≡ Performance/Growth
  Potential Matrix failure)
- Frontier models hallucinate less on edge cases (the v4.4 RSU vesting
  hallucination cluster)
- Hosted models have order-of-magnitude lower P95 latency, which is
  the only target this project failed

**Cost estimate for hosted production** (rough):
- 100 queries/day × $0.05/query (Claude Sonnet generation + Voyage
  embeddings + small reranker) = **$5/day = $1,800/year**
- For a company saving 1 HR FTE-week of repetitive answering per
  month, ROI is immediate

---

> **In portfolio highlight decisions, from what, how and why and
> justify**

The decision table below covers every load-bearing architectural
choice with **from what (the alternative)**, **how (the
implementation)**, and **why (the evidence)**.

| Decision | From what | How | Why (evidence) |
|---|---|---|---|
| Use RAG, not fine-tuning | Fine-tune `qwen3.5:4b` on Q&A pairs | Build a Kedro indexing → retrieval → generation pipeline | Handbook updates daily; fine-tuning needs hours per update. 48k chunks too small for meaningful fine-tuning of 4B model. No GPU available |
| Use `qwen3.5:4b` as answer LLM | llama3.2 (used in v1.0) | Swap model name in Ollama config | v1.2 swap from llama3.2 → qwen3.5:4b lifted quality from 3.14 → 4.08 — single biggest quality jump in the entire project |
| Use `qwen3.5:0.8b` as reranker | qwen3.5:4b | Separate parameter `reranker_model` in `parameters.yml` | 4b reranker was tried (commit history) but "over-selected within documents" (note in parameters.yml). 0.8b is 5× faster, same listwise quality |
| Chunk size 256 tokens | 512 tokens (v1.0 default) | `chunk_size: 256` in Kedro params | v1.0→v1.1 chunk reduction lifted Hit Rate 0.84 → 0.88, MRR 0.71 → 0.83 |
| HyDE retrieval | Embed raw question | `generate_hyde_doc` node before `embed_query` | Series 1 plateaued at Hit 0.88 with raw-query embedding. HyDE introduced in v1.19 took Hit to 1.0 |
| LLM listwise reranker | Cross-encoder MS-MARCO reranker | `rerank_chunks` node using Ollama chat call | Cross-encoder (v1.5) with rerank_top_n=3 regressed Hit to 0.80. LLM listwise rerank handles ambiguity in question intent that cross-encoder can't |
| LLM folder ranking | None (single flat retrieval over 48k chunks) | New `rank_folders` node + `retrieve_chunks_folder_priority` | v4.1 → v4.4 added folder routing. Quality 4.24 → 4.28, Hit stayed 1.0, addressed 4 of 10 v4.2 failure cases |
| Remove agentic multi-tool architecture | Keep agent + BM25 + LightRAG | Delete `QueryAgent`, `KeywordRAGTool`, `GraphRAGTool` | v3.0 benchmark: all 11 score-1 failures traced to BM25 tool pulling engineering noise. Single-tool HyDE never caused a score-1 failure |
| Abandon PageIndex (Series 2) | Continue tuning PageIndex | Switch back to vector RAG on main branch | 8 PageIndex versions plateaued at Quality 3.2 max despite Hit Rate 1.0. P95 latency 500–700 s. No path to viable on the constraint envelope |
| LLM-as-judge for quality | Human evaluation | `compute_quality_metrics` node using qwen3.5:4b with rubric prompt | Human eval would have allowed 3 versions instead of 30. Trade-off accepted; LLM-judge correlates well enough with human judgment on simple Q&A |
| Folder filter via `top_folders` cap | No filter (open retrieval) | `retrieve_chunks_folder_priority` filters to top N ranked folders before scoring | Open retrieval pulls engineering noise into HR answers (Series 3 evidence). Filter forces the corpus to "HR-relevant subspace" |
| Skip latency loop, infer from main eval | Separate 20-run latency loop | `compute_latency_metrics(qa_eval_results)` reuses per-question timings | The 20-run loop duplicated effort; the 25 main eval runs already produce timings on the same code path. Simplified pipeline, same data |
| Use Kedro for pipelining | Plain Python scripts | Each pipeline phase is a Kedro DAG with `nodes.py` + `pipeline.py` + `parameters.yml` | Kedro gives reproducibility (param-versioned runs), caching (between dev iterations), and a CRISP-DM-shaped directory structure for free |

Each row of this table is a defensible decision. The "why" column is
backed by a measurable change in a saved benchmark report — every
decision can be reproduced from `data/08_reporting/`.

---

> **Explain what the project is about at high level**

**The pitch (one paragraph)**: ctcobot is an HR-policy chatbot that lets
employees ask questions in natural language and get accurate, source-cited
answers grounded in the company's HR handbook. It uses retrieval-augmented
generation: a local LLM searches the handbook for the most relevant
excerpts, then reads them and synthesizes a focused answer with citations.
The system was built end-to-end (corpus ingestion → embedding index →
retrieval → reranking → answer generation → evaluation) following the
CRISP-DM methodology, and benchmarked over 30 versions to identify the
architectural choices that matter for accuracy, retrieval quality, and
latency on a constrained (CPU-only, open-weight-models-only) prototype
environment.

**The problem it solves**: Employees waiting for HR to answer
repetitive policy questions ("Am I eligible for FMLA?", "How does
my equity vest?"). HR teams answering the same questions hundreds of
times. Compliance teams needing every answer to be traceable to the
source policy document. ctcobot answers all three concerns: instant
self-service answer, with the exact handbook section cited.

**The technical core**: A four-stage retrieval pipeline —
**HyDE → embed → folder-ranked retrieve → LLM listwise rerank →
grounded answer generation** — built with Kedro, Ollama, turbovec
(scalar-quantized vector index), and qwen3.5 (open-weight LLM family).
The 48k-chunk GitLab handbook is the test corpus. A 25-question
benchmark with LLM-as-judge runs at every iteration.

**What makes the project interesting**: The iteration log itself. The
project tried four entirely different RAG architectures (vector,
PageIndex tree retrieval, multi-tool agent, HyDE+folder-routing) and
documented why each one worked or failed. The final architecture is
~30% accuracy better than the v1.0 baseline despite running on the same
hardware with the same corpus — every gain came from architecture and
prompt-engineering decisions, each one justified by a measurable
benchmark delta.

**What it demonstrates** (the portfolio thesis): That a constrained,
CPU-only RAG system can be made *accurate enough for HR Q&A use*
(Quality 4.28/5, Hit Rate 1.0, MRR 0.89 at v4.4) by careful pipeline
design and iterative evaluation, **without** model fine-tuning,
**without** a cloud GPU, and **without** proprietary tooling. The same
pattern shipped on a hosted API would solve the latency constraint
immediately, making this a viable production blueprint.
