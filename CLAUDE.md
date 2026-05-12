# ctcobot — Version History

All benchmark runs are stored in `data/08_reporting/`. Targets: hit_rate ≥ 0.70, MRR ≥ 0.55, avg_quality ≥ 3.5, P95 latency ≤ 5s.

---

## Series 1 — Vectorized RAG (`main` branch)

Architecture: embed query → cosine-similarity retrieval from FAISS index → optional cross-encoder rerank → LLM answer generation.

### v1.0 — Baseline (2026-03-20)
**File:** `benchmark_report_v1.json`
| Metric | Value |
|--------|-------|
| Quality | 3.19 |
| Hit Rate | 0.84 |
| MRR | 0.707 |
| P95 Latency | 46.3s |

**Config:** chunk_size=512, chunk_overlap=50, embedding=nomic-embed-text, llm=llama3.2, judge=llama3.2, top_k=5. No reranker.

---

### v1.1 — Smaller chunks (2026-03-27)
**File:** `benchmark_report_v1-1.json`
| Metric | Value |
|--------|-------|
| Quality | 3.14 |
| Hit Rate | 0.88 |
| MRR | 0.827 |
| P95 Latency | 34.4s |

**Changes from v1.0:**
- `chunk_size` 512 → 256 (better granularity per chunk)

---

### v1.2 — Stronger LLM (2026-03-27)
**File:** `benchmark_report_v1-2.json`
| Metric | Value |
|--------|-------|
| Quality | 4.08 |
| Hit Rate | 0.88 |
| MRR | 0.827 |
| P95 Latency | 98.2s |

**Changes from v1.1:**
- llm: llama3.2 → qwen3.5:4b
- judge: llama3.2 → qwen3.5:4b
- Large quality gain but latency doubled.

---

### v1.3 — Smaller LLM, smaller judge (2026-03-27)
**File:** `benchmark_report_v1-3.json`
| Metric | Value |
|--------|-------|
| Quality | 3.08 |
| Hit Rate | 0.88 |
| MRR | 0.827 |
| P95 Latency | 32.3s |

**Changes from v1.2:**
- llm: qwen3.5:4b → qwen3.5:0.8b
- judge: qwen3.5:4b → qwen3.5:0.8b
- Quality dropped significantly — weaker judge scores more harshly.

---

### v1.4 — Strong judge, small LLM (2026-03-27)
**File:** `benchmark_report_v1-4.json`
| Metric | Value |
|--------|-------|
| Quality | 3.52 |
| Hit Rate | 0.88 |
| MRR | 0.827 |
| P95 Latency | 32.1s |

**Changes from v1.3:**
- judge: qwen3.5:0.8b → qwen3.5:4b (restored)
- llm stays qwen3.5:0.8b for speed
- Best quality/latency trade-off so far.

---

### v1.5 — Cross-encoder reranker added (2026-03-30)
**File:** `benchmark_report_v1-5.json`
| Metric | Value |
|--------|-------|
| Quality | 3.24 |
| Hit Rate | 0.80 |
| MRR | 0.780 |
| P95 Latency | 27.0s |

**Changes from v1.4:**
- Added reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`, rerank_top_n=3
- Retrieval metrics regressed — reranker with top_n=3 discards too many candidates.

---

### v1.6 — Higher top_k and rerank_top_n (2026-03-30)
**File:** `benchmark_report_v1-6.json`
| Metric | Value |
|--------|-------|
| Quality | 3.92 |
| Hit Rate | 0.88 |
| MRR | 0.808 |
| P95 Latency | 28.7s |

**Changes from v1.5:**
- top_k: 5 → 10
- rerank_top_n: 3 → 5
- Best quality in series so far; retrieval recovered.

---

### v1.7 — Even higher top_k (2026-03-30)
**File:** `benchmark_report_v1-7.json`
| Metric | Value |
|--------|-------|
| Quality | 3.56 |
| Hit Rate | 0.84 |
| MRR | 0.773 |
| P95 Latency | 35.8s |

**Changes from v1.6:**
- top_k: 10 → 15
- Quality and retrieval both regressed vs v1.6 — more noise at top_k=15.

---

### v1.8 — Upgraded answer LLM (2026-03-31)
**File:** `benchmark_report_v1-8.json`
| Metric | Value |
|--------|-------|
| Quality | 4.12 |
| Hit Rate | 0.88 |
| MRR | 0.808 |
| P95 Latency | 90.2s |

**Changes from v1.6** (reverted top_k=15 → 10):
- llm: qwen3.5:0.8b → qwen3.5:4b
- top_k back to 10
- Highest quality in Series 1 but P95 latency tripled.

---

### v1.9 — Semantic chunking, large max_tokens (2026-04-07)
**File:** `benchmark_report_v1-9.json`
| Metric | Value |
|--------|-------|
| Quality | 3.68 |
| Hit Rate | 0.84 |
| MRR | 0.763 |
| P95 Latency | 36.7s |

**Changes from v1.6** (baseline config for semantic chunking experiment):
- Replaced fixed-size chunking with semantic chunking (all-MiniLM-L6-v2, percentile breakpoint at 95th)
- max_chunk_tokens=1024 (large chunks)
- llm: back to qwen3.5:0.8b for speed
- Retrieval metrics lower than v1.6.

---

### v1.10 — Semantic chunking, small max_tokens (2026-04-08)
**File:** `benchmark_report_v1-10.json`
| Metric | Value |
|--------|-------|
| Quality | 3.16 |
| Hit Rate | 0.84 |
| MRR | 0.793 |
| P95 Latency | 24.0s |

**Changes from v1.9:**
- max_chunk_tokens: 1024 → 256
- Quality dropped — smaller semantic chunks lost context.

---

### v1.11 — Tightened SYSTEM_PROMPT (2026-04-08)
**File:** `benchmark_report_v1-11.json`
| Metric | Value |
|--------|-------|
| Quality | 3.60 |
| Hit Rate | 0.88 |
| MRR | 0.833 |
| P95 Latency | 26.5s |

**Changes from v1.10:**
- Tightened SYSTEM_PROMPT: "I could not find" now only allowed when **every** excerpt is unrelated; partial matches must be extracted and stated.
- Quality improved significantly (+0.44) from prompt change alone.

---

### v1.12 — Re-run of v1.11 config (2026-04-10)
**File:** `benchmark_report_v1-12.json`
| Metric | Value |
|--------|-------|
| Quality | 3.36 |
| Hit Rate | 0.88 |
| MRR | 0.833 |
| P95 Latency | 27.3s |

**Changes from v1.11:** None (same parameters). Score variation is due to LLM non-determinism.
This is the **last vectorized RAG version** before switching to PageIndex.

---

## Series 2 — PageIndex RAG (`PageIndex-ver` branch)

Architecture: LLM-driven 3-stage retrieval using hierarchical document tree (PageIndex):
1. Stage 1 — LLM scans document registry, selects top-N documents
2. Stage 2 — LLM navigates each document's tree, selects relevant sections
3. Stage 3 — Listwise LLM reranker selects final chunks

All versions use: embedding=none, llm=qwen3.5:4b, judge=qwen3.5:4b, pageindex_model=qwen3.5:4b.

---

### v2.0 — First PageIndex run (2026-04-14)
**File:** `benchmark_report_v3-0.json` (internal version 2.0)
| Metric | Value |
|--------|-------|
| Quality | 2.28 |
| Hit Rate | 0.48 |
| MRR | 0.353 |
| P95 Latency | 183s |

**Config:** pageindex_top_docs=5, pageindex_top_sections=3, llm=qwen3.5:0.8b
Very poor retrieval — too few docs and sections selected.

---

### v2.1 — More docs/sections, stronger LLM (2026-05-01)
**File:** `benchmark_report_v3-1.json` (internal version 2.1)
| Metric | Value |
|--------|-------|
| Quality | 3.20 |
| Hit Rate | 0.60 |
| MRR | 0.463 |
| P95 Latency | 348s |

**Changes from v2.0:**
- llm: qwen3.5:0.8b → qwen3.5:4b
- pageindex_top_docs: 5 → 8
- pageindex_top_sections: 3 → 5
- Quality improved substantially; retrieval still weak.

---

### v2.2 — Retrieval logic improvements (2026-05-04)
**File:** `benchmark_report_v3-2.json` (internal version 2.2)
| Metric | Value |
|--------|-------|
| Quality | 3.56 |
| Hit Rate | 0.68 |
| MRR | 0.425 |
| P95 Latency | 474s |

**Changes from v2.1:** Code changes to retrieval logic (exact changes not recorded). Quality improved.

---

### v2.3 — More sections per doc (2026-05-05)
**File:** `benchmark_report_v3-3.json` (internal version 2.3)
| Metric | Value |
|--------|-------|
| Quality | 2.96 |
| Hit Rate | 1.00 |
| MRR | 0.689 |
| P95 Latency | 611s |

**Changes from v2.2:**
- pageindex_top_sections: 5 → 8
- Hit rate jumped to 1.0 (correct doc always retrieved) but quality dropped — too many chunks overwhelm context.

---

### v2.4 — Sections reverted, code refactor (2026-05-05)
**File:** `benchmark_report_v2-4.json` (internal version 2.4)
| Metric | Value |
|--------|-------|
| Quality | 2.48 |
| Hit Rate | 1.00 |
| MRR | 0.569 |
| P95 Latency | 736s |

**Changes from v2.3:**
- pageindex_top_sections: 8 → 5
- Code refactor (exact changes not recorded). Quality regressed further.

---

### v2.5 — Major retrieval improvements (2026-05-05)
**File:** `benchmark_report_v2-5.json` (internal version 2.5)
| Metric | Value |
|--------|-------|
| Quality | 3.20 |
| Hit Rate | 1.00 |
| MRR | 0.704 |
| P95 Latency | 607s |

**Changes from v2.4:**
- Significant retrieval improvements (exact code changes not recorded).
- Quality jumped +0.72. Best PageIndex quality at this point.

---

### v2.6 — Fix attempts (quality regression) (2026-05-06)
**File:** `benchmark_report_v2-6.json` (internal version 2.6)
| Metric | Value |
|--------|-------|
| Quality | 2.60 |
| Hit Rate | 1.00 |
| MRR | 0.649 |
| P95 Latency | 510s |

**Changes from v2.5:**
- Pre-reranker per-source chunk cap (max 2 chunks per source, applied before reranking)
- Stage 2 section selection changed to use original `question` instead of `expanded_question`
- Reranker preview: 300 → 600 chars
- Tightened SYSTEM_PROMPT "I could not find" condition

**Root cause of regression:** Using original `question` for Stage 2 broke acronym/synonym expansion needed for section navigation (e.g., "TNTR" not expanded to "Too New To Rate", "9-box" not expanded to "performance potential matrix").

---

### v2.7 — Fix 4 reverted, post-rank diversity (2026-05-06)
**File:** `benchmark_report_v2-7.json` (internal version 2.7)
| Metric | Value |
|--------|-------|
| Quality | 2.48 |
| Hit Rate | 0.96 |
| MRR | 0.596 |
| P95 Latency | 694s |

**Changes from v2.6:**
- Reverted Stage 2 to use `expanded_question` (Fix 4 undone)
- Per-source chunk cap moved to **after** reranking (post-rank diversity: top-2 per source in relevance order)
- Reranker pulls 3× candidates (pageindex_reranker_top_k × 3) before diversity filter
- pageindex_top_sections: 5 → 8
- Quality did not recover — post-rank diversity + top_sections=8 introduced new issues.

**Note:** P95 latency is physically unreachable on local Ollama (~12 sequential LLM calls × ~50s each). The 5s target only applies to production API deployments (e.g. Claude API).

---

## Summary Table

| Version | Quality | Hit Rate | MRR | P95 (s) | Key Change |
|---------|---------|----------|-----|---------|------------|
| v1.0 | 3.19 | 0.84 | 0.707 | 46 | Baseline (llama3.2, chunk=512, top_k=5) |
| v1.1 | 3.14 | 0.88 | 0.827 | 34 | chunk_size 512→256 |
| v1.2 | 4.08 | 0.88 | 0.827 | 98 | llm/judge: llama3.2→qwen3.5:4b |
| v1.3 | 3.08 | 0.88 | 0.827 | 32 | llm/judge: →qwen3.5:0.8b |
| v1.4 | 3.52 | 0.88 | 0.827 | 32 | judge: →qwen3.5:4b, llm stays 0.8b |
| v1.5 | 3.24 | 0.80 | 0.780 | 27 | Added cross-encoder reranker, rerank_top_n=3 |
| v1.6 | 3.92 | 0.88 | 0.808 | 29 | top_k 5→10, rerank_top_n 3→5 |
| v1.7 | 3.56 | 0.84 | 0.773 | 36 | top_k 10→15 (regressed) |
| v1.8 | 4.12 | 0.88 | 0.808 | 90 | llm: →qwen3.5:4b, top_k back to 10 |
| v1.9 | 3.68 | 0.84 | 0.763 | 37 | Semantic chunking (max_tokens=1024) |
| v1.10 | 3.16 | 0.84 | 0.793 | 24 | Semantic chunking max_tokens 1024→256 |
| v1.11 | 3.60 | 0.88 | 0.833 | 27 | Tightened SYSTEM_PROMPT "I could not find" |
| v1.12 | 3.36 | 0.88 | 0.833 | 27 | Re-run of v1.11 (LLM variance) |
| v2.0 | 2.28 | 0.48 | 0.353 | 183 | First PageIndex run (top_docs=5, top_sections=3) |
| v2.1 | 3.20 | 0.60 | 0.463 | 348 | top_docs=8, top_sections=5, llm→4b |
| v2.2 | 3.56 | 0.68 | 0.425 | 474 | Retrieval logic improvements |
| v2.3 | 2.96 | 1.00 | 0.689 | 611 | top_sections 5→8 |
| v2.4 | 2.48 | 1.00 | 0.569 | 736 | top_sections 8→5, code refactor |
| v2.5 | 3.20 | 1.00 | 0.704 | 607 | Major retrieval improvements |
| v2.6 | 2.60 | 1.00 | 0.649 | 510 | Pre-rank cap + original question for Stage 2 (regression) |
| v2.7 | 2.48 | 0.96 | 0.596 | 694 | Post-rank diversity, expanded_question restored, top_sections=8 |
