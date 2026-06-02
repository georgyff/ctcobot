"""
Agentic query router for v3.0.

QueryAgent uses Ollama's tool-calling API to decide which RAG tools to invoke
(vector_rag, graph_rag, keyword_rag), executes them, then synthesizes a final
answer from the combined results.
"""
import json
import logging

import requests

from ctcobot.prompt_templates import SYSTEM_PROMPT, format_context, format_rag_prompt

logger = logging.getLogger(__name__)

AGENT_SYSTEM_PROMPT = (
    "You are ctcobot, an HR policy assistant for a tech company. "
    "You have access to two search tools. Use them to find relevant policy information:\n\n"
    "- vector_rag: Semantic vector search using HyDE. Best for conceptual questions about "
    "policies, procedures, and general HR topics.\n"
    "- keyword_rag: BM25 keyword matching. Best for specific terms, acronyms (PTO, FMLA, "
    "SOC 2), policy codes, and exact phrase lookups.\n\n"
    "You may call multiple tools if the question spans multiple domains.\n\n"
    "When answering after using tools:\n"
    "- Use ONLY the information retrieved by the tools. Do not draw on outside knowledge.\n"
    "- Only say 'I could not find information about this in the company handbook.' when "
    "ALL retrieved excerpts are completely unrelated to the question. If any excerpt "
    "contains partial information, extract and state what is there.\n"
    "- Be factual, concise, and professional."
)

AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "vector_rag",
            "description": (
                "Semantic vector search using HyDE (Hypothetical Document Embedding). "
                "Best for conceptual questions about policies, procedures, and general HR topics."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The question to search for"},
                },
                "required": ["query"],
            },
        },
    },
    # graph_rag disabled until LightRAG index is built
    # {
    #     "type": "function",
    #     "function": {
    #         "name": "graph_rag",
    #         "description": (
    #             "Knowledge graph search for entity relationships and cross-policy connections. "
    #             "Best for questions about how policies relate to each other, organizational "
    #             "structure, or multi-policy questions. "
    #             "mode: local=specific entity facts, global=high-level themes, mix=both (default)."
    #         ),
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "query": {"type": "string"},
    #                 "mode": {
    #                     "type": "string",
    #                     "enum": ["local", "global", "mix"],
    #                     "description": "Search mode: local, global, or mix (default)",
    #                 },
    #             },
    #             "required": ["query"],
    #         },
    #     },
    # },
    {
        "type": "function",
        "function": {
            "name": "keyword_rag",
            "description": (
                "BM25 keyword matching over the full document corpus. "
                "Best for specific terms, acronyms (PTO, FMLA, SOC 2), policy codes, "
                "and exact phrase lookups."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
            },
        },
    },
]

MAX_TOOL_ROUNDS = 3


class QueryAgent:
    """Agentic query router that selects and invokes RAG tools via Ollama tool calling."""

    def __init__(
        self,
        tools: dict,
        ollama_base_url: str,
        agent_model: str,
        llm_model: str,
    ):
        self.tools = tools  # {"vector_rag": VectorRAGTool(), ...}
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.agent_model = agent_model
        self.llm_model = llm_model

    def run(self, question: str) -> dict:
        """Run the agentic loop and return the final answer dict.

        Returns:
            Dict with keys: question, answer, sources, tools_used, chunks,
            pre_rerank_sources (from VectorRAGTool if used).
        """
        messages = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        tools_used: list[str] = []
        all_chunks: list[dict] = []
        message: dict = {}

        for _ in range(MAX_TOOL_ROUNDS):
            response_data = self._call_llm_with_tools(messages)
            message = response_data.get("message", {})
            tool_calls = message.get("tool_calls") or []

            if not tool_calls:
                break

            messages.append(
                {
                    "role": "assistant",
                    "content": message.get("content", ""),
                    "tool_calls": tool_calls,
                }
            )

            for tc in tool_calls:
                fn = tc.get("function", {})
                tool_name = fn.get("name", "")
                tool_args = fn.get("arguments", {})
                if isinstance(tool_args, str):
                    try:
                        tool_args = json.loads(tool_args)
                    except json.JSONDecodeError:
                        tool_args = {}

                chunks = self._execute_tool(tool_name, tool_args)
                tools_used.append(tool_name)
                all_chunks.extend(chunks)

                messages.append(
                    {
                        "role": "tool",
                        "content": json.dumps(chunks),
                        "name": tool_name,
                    }
                )

        # Deduplicate chunks across tools, keeping highest score
        seen: dict[tuple, dict] = {}
        for chunk in all_chunks:
            key = (chunk.get("source_path", ""), chunk.get("text", "")[:80])
            if key not in seen or chunk.get("score", 0.0) > seen[key].get("score", 0.0):
                seen[key] = chunk
        deduped = sorted(seen.values(), key=lambda c: c.get("score", 0.0), reverse=True)

        # Reuse the agent's final response when it already synthesized from tool results.
        # The last _call_llm_with_tools call (where tool_calls=[]) already has full context
        # of tool results and produced a grounded answer — no need for a second LLM call.
        final_content = message.get("content", "").strip()
        if final_content and all_chunks:
            answer = final_content
        else:
            answer = self._synthesize_answer(question, deduped)

        seen_srcs: set[str] = set()
        sources = []
        for chunk in deduped:
            src = chunk.get("source_path", "unknown")
            if src not in seen_srcs:
                seen_srcs.add(src)
                sources.append({"source_path": src, "score": chunk.get("score", 0.0)})

        # Capture pre-rerank sources from VectorRAGTool if it was used
        vector_tool = self.tools.get("vector_rag")
        pre_rerank_sources = (
            vector_tool.last_pre_rerank_sources
            if vector_tool and "vector_rag" in tools_used
            else []
        )

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "tools_used": list(dict.fromkeys(tools_used)),
            "chunks": deduped,
            "pre_rerank_sources": pre_rerank_sources,
        }

    def _call_llm_with_tools(self, messages: list[dict]) -> dict:
        resp = requests.post(
            f"{self.ollama_base_url}/api/chat",
            json={
                "model": self.agent_model,
                "messages": messages,
                "tools": AGENT_TOOLS,
                "stream": False,
                "options": {"think": False},
            },
            timeout=500,
        )
        resp.raise_for_status()
        return resp.json()

    def _execute_tool(self, tool_name: str, tool_args: dict) -> list[dict]:
        tool = self.tools.get(tool_name)
        if tool is None:
            logger.warning("Unknown tool requested: %s", tool_name)
            return []
        try:
            result = tool(**tool_args)
            logger.info("Tool %s returned %d chunks", tool_name, len(result))
            return result
        except Exception as e:
            logger.warning("Tool %s failed: %s", tool_name, e)
            return []

    def _synthesize_answer(self, question: str, chunks: list[dict]) -> str:
        if not chunks:
            return "I could not find information about this in the company handbook."

        context = format_context(chunks)
        user_prompt = format_rag_prompt(question, context)

        resp = requests.post(
            f"{self.ollama_base_url}/api/chat",
            json={
                "model": self.llm_model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "options": {"think": False},
            },
            timeout=180,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()
