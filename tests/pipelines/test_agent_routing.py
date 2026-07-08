"""
Unit tests for the agent's tool-selection logic (pure, no LLM).
"""
from ctcobot.pipelines.querying.agent import _question_needs_keyword, _select_tools

ALL_TOOLS = {"vector_rag": object(), "keyword_rag": object(), "graph_rag": object()}
CONCEPTUAL_Q = "How do performance reviews work?"
ACRONYM_Q = "What does TNTR stand for?"


class TestSelectTools:
    def test_router_choice_is_authoritative_no_vector_floor(self):
        assert _select_tools(["keyword_rag"], ALL_TOOLS, CONCEPTUAL_Q) == ["keyword_rag"]
        assert _select_tools(["graph_rag"], ALL_TOOLS, CONCEPTUAL_Q) == ["graph_rag"]

    def test_multi_tool_routing_preserved_in_order(self):
        assert _select_tools(
            ["graph_rag", "vector_rag"], ALL_TOOLS, CONCEPTUAL_Q
        ) == ["graph_rag", "vector_rag"]

    def test_empty_routing_falls_back_to_vector(self):
        assert _select_tools([], ALL_TOOLS, CONCEPTUAL_Q) == ["vector_rag"]

    def test_empty_routing_without_vector_uses_first_registered(self):
        tools = {"keyword_rag": object(), "graph_rag": object()}
        assert _select_tools([], tools, CONCEPTUAL_Q) == ["keyword_rag"]

    def test_unregistered_names_filtered_then_fallback(self):
        assert _select_tools(["bogus_tool"], ALL_TOOLS, CONCEPTUAL_Q) == ["vector_rag"]

    def test_acronym_net_adds_keyword_to_routed_choice(self):
        assert _select_tools(["vector_rag"], ALL_TOOLS, ACRONYM_Q) == [
            "vector_rag", "keyword_rag",
        ]

    def test_acronym_net_does_not_duplicate_keyword(self):
        assert _select_tools(["keyword_rag"], ALL_TOOLS, ACRONYM_Q) == ["keyword_rag"]

    def test_routing_failure_on_acronym_question_gets_vector_and_keyword(self):
        # Router returned nothing on a literal-term question: the fallback
        # fires first (vector), then the net adds keyword — a routing FAILURE
        # never degrades to keyword-only retrieval. A deliberate keyword-only
        # pick (see test above) is still respected.
        assert _select_tools([], ALL_TOOLS, ACRONYM_Q) == [
            "vector_rag", "keyword_rag",
        ]

    def test_always_at_least_one_tool(self):
        for routed in ([], ["bogus"], ["vector_rag"], ["graph_rag"]):
            assert len(_select_tools(routed, ALL_TOOLS, CONCEPTUAL_Q)) >= 1


class TestQuestionNeedsKeyword:
    def test_acronym_and_hyphen_terms_trigger(self):
        assert _question_needs_keyword(ACRONYM_Q)
        assert _question_needs_keyword("How does the 9-box matrix work?")

    def test_plain_question_does_not(self):
        assert not _question_needs_keyword(CONCEPTUAL_Q)
