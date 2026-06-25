"""
Agentic RAG router.

A QueryAgent uses an Ollama tool-calling LLM to decide which retrieval tool(s)
to invoke for a given question, executes them, merges and reranks the results,
then synthesizes an answer.

Two tools are wired in:
  - vector_rag  (VectorRAGTool)  — semantic HyDE retrieval
  - keyword_rag (KeywordRAGTool) — folder-scoped BM25 lexical retrieval
"""
import logging
import re

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 3

# Bound the routing LLM call; on timeout the agent falls back to vector_rag.
ROUTE_TIMEOUT_SECONDS = 60

# Heuristic safety net: questions hinging on an acronym, a quoted phrase, or a
# hyphenated alphanumeric term (e.g. TNTR, "Too New To Rate", 9-box) should
# always reach BM25 even if the LLM router does not pick it. This directly
# targets the documented exact-term failure class.
_ACRONYM_RE = re.compile(r"\b[A-Z]{2,}\b")
_QUOTED_RE = re.compile(r"[\"'].+?[\"']")
# Hyphenated tokens like "9-box" / "v1-2"; flagged when they mix digit + letter.
_HYPHEN_TOKEN_RE = re.compile(r"\b[A-Za-z0-9]+-[A-Za-z0-9]+\b")


def _question_needs_keyword(question: str) -> bool:
    """True if the question contains a literal term BM25 is well suited to."""
    if _ACRONYM_RE.search(question) or _QUOTED_RE.search(question):
        return True
    for tok in _HYPHEN_TOKEN_RE.findall(question):
        has_digit = any(c.isdigit() for c in tok)
        has_alpha = any(c.isalpha() for c in tok)
        if has_digit and has_alpha:
            return True
    return False


class QueryAgent:
    """Routes a question to vector_rag and/or keyword_rag, then answers.

    Flow:
      1. rank_folders once, shared by both tools (avoids duplicate LLM calls).
      2. LLM tool-calling round(s) choose the tool(s).
      3. Acronym safety net force-includes keyword_rag when warranted.
      4. Execute chosen tools, passing the shared folder ranking.
      5. Merge + dedupe chunks, LLM listwise rerank, generate the answer.
    """

    def __init__(
        self,
        tools: dict,
        ollama_base_url: str,
        agent_model: str,
        llm_model: str,
        reranker_model: str,
        rerank_top_n: int,
        turbovec_persist_path: str,
        agent_reranker_model: str | None = None,
    ):
        self.tools = tools
        self.ollama_base_url = ollama_base_url
        self.agent_model = agent_model
        self.llm_model = llm_model
        self.reranker_model = reranker_model
        # The merged set concentrates cross-folder BM25 noise, so it gets a
        # stronger reranker than the per-tool default when configured.
        self.agent_reranker_model = agent_reranker_model or reranker_model
        self.rerank_top_n = rerank_top_n
        self.turbovec_persist_path = turbovec_persist_path
        self.last_tools_used: list[str] = []
        self.last_pre_rerank_sources: list[str] = []
        self.last_pre_rerank_chunks: list[dict] = []

    def _route(self, question: str) -> list[str]:
        """Ask the LLM which tool(s) to use; return an ordered list of names."""
        import ollama
        from ctcobot.prompt_templates import AGENT_SYSTEM_PROMPT, AGENT_TOOLS

        client = ollama.Client(host=self.ollama_base_url, timeout=ROUTE_TIMEOUT_SECONDS)
        messages = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        chosen: list[str] = []
        try:
            for _ in range(MAX_TOOL_ROUNDS):
                response = client.chat(
                    model=self.agent_model,
                    messages=messages,
                    tools=AGENT_TOOLS,
                    think=False,
                )
                tool_calls = response["message"].get("tool_calls") or []
                if not tool_calls:
                    break
                for tc in tool_calls:
                    name = tc["function"]["name"]
                    if name in self.tools and name not in chosen:
                        chosen.append(name)
                # One routing round is enough for our two-tool setup.
                break
        except Exception as e:
            logger.warning("Agent routing failed (%s); defaulting to vector_rag", e)

        return chosen

    def run(self, question: str) -> dict:
        """Route, retrieve, merge, rerank, and generate an answer."""
        from ctcobot.pipelines.querying.nodes import (
            rank_folders,
            rerank_chunks,
            build_prompt,
            generate_answer,
        )

        # 1. Shared folder ranking — one LLM call for both tools.
        ranked_folders = rank_folders(
            question, self.turbovec_persist_path, self.ollama_base_url, self.llm_model,
        )

        # 2. LLM routing decides which optional tools to add to the vector floor.
        routed = self._route(question)

        # 3. vector_rag is ALWAYS run as the semantic floor; the router and the
        #    acronym safety net only decide whether to ALSO run keyword_rag.
        #    This prevents the keyword-only catastrophic misses seen in v5.0
        #    (9-box, TNTR), turning BM25 into a pure recall booster.
        chosen: list[str] = []
        if "vector_rag" in self.tools:
            chosen.append("vector_rag")

        want_keyword = "keyword_rag" in routed or _question_needs_keyword(question)
        if want_keyword and "keyword_rag" in self.tools and "keyword_rag" not in chosen:
            chosen.append("keyword_rag")

        # 4. Fallback: if vector_rag isn't registered, use whatever routed (or
        #    the first available tool) so we never run with no tool.
        if not chosen:
            chosen = routed or list(self.tools)[:1]
        logger.info("Agent routing chose: %s | %s", chosen, question[:60])

        # 5. Execute tools, sharing the folder ranking. Collect the deduped
        #    pre-rerank candidate set (by source+chunk) for clean retrieval
        #    metrics, plus the merged chunks for answer synthesis.
        merged: dict[tuple, dict] = {}
        pre_rerank_chunks: dict[tuple, dict] = {}
        for name in chosen:
            tool = self.tools[name]
            chunks = tool(question, ranked_folders=ranked_folders)
            for c in getattr(tool, "last_pre_rerank_chunks", []):
                pre_rerank_chunks.setdefault((c["source_path"], c["chunk_index"]), c)
            for c in chunks:
                key = (c["source_path"], c["chunk_index"])
                if key not in merged or c["score"] > merged[key]["score"]:
                    merged[key] = c

        deduped_pre_rerank = list(pre_rerank_chunks.values())
        self.last_tools_used = chosen
        self.last_pre_rerank_chunks = deduped_pre_rerank
        self.last_pre_rerank_sources = [c["source_path"] for c in deduped_pre_rerank]

        candidates = list(merged.values())
        if not candidates:
            logger.warning("Agent retrieved no chunks for: %s", question[:60])

        # 6. Rerank the merged candidate set (stronger reranker — this is where
        #    cross-folder BM25 noise concentrates) and synthesize the answer.
        reranked = rerank_chunks(
            candidates, question, self.agent_reranker_model,
            self.rerank_top_n, self.ollama_base_url,
        )
        prompt_data = build_prompt(question, reranked)
        result = generate_answer(prompt_data, self.ollama_base_url, self.llm_model)
        result["tools_used"] = chosen
        result["pre_rerank_sources"] = self.last_pre_rerank_sources
        result["pre_rerank_chunks"] = deduped_pre_rerank
        return result
