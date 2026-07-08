"""
Unit tests for the LightRAG graph-tool integration (pure logic only).

Nothing here touches Ollama, lightrag storages, or the network — the
LLM-dependent paths are exercised through their pure mapping/gating helpers.
"""
import pytest

from ctcobot.pipelines.querying.tools import (
    _GRAPH_CHUNK_INDEX_BASE,
    _GRAPH_KG_SOURCE,
    _graph_data_to_chunks,
    _graph_store_ready,
    maybe_graph_tool,
)


def _payload(entities=(), relationships=(), chunks=(), status="success"):
    return {
        "status": status,
        "data": {
            "entities": list(entities),
            "relationships": list(relationships),
            "chunks": list(chunks),
        },
    }


class TestGraphDataToChunks:
    def test_failure_payload_yields_nothing(self):
        assert _graph_data_to_chunks({"status": "failure", "data": {}}, 5) == []
        assert _graph_data_to_chunks(None, 5) == []

    def test_doc_chunks_carry_real_source_metadata(self):
        payload = _payload(chunks=[
            {"content": "PTO policy text", "file_path": "people-group/pto.md"},
        ])
        (chunk,) = _graph_data_to_chunks(payload, 5)
        assert chunk["source_path"] == "people-group/pto.md"
        assert chunk["folder"] == "people-group"
        assert chunk["filename"] == "pto.md"
        assert chunk["chunk_index"] == _GRAPH_CHUNK_INDEX_BASE
        assert chunk["text"] == "PTO policy text"

    def test_unknown_source_chunks_fall_back_to_graph_namespace(self):
        payload = _payload(chunks=[
            {"content": "orphan text", "file_path": "unknown_source"},
        ])
        (chunk,) = _graph_data_to_chunks(payload, 5)
        assert chunk["source_path"] == _GRAPH_KG_SOURCE
        assert chunk["folder"] == "__graph__"

    def test_chunk_index_namespace_never_collides_with_turbovec(self):
        payload = _payload(chunks=[
            {"content": f"c{i}", "file_path": "legal/x.md"} for i in range(3)
        ])
        out = _graph_data_to_chunks(payload, 5)
        assert all(c["chunk_index"] >= _GRAPH_CHUNK_INDEX_BASE - 1 for c in out)

    def test_max_chunks_cap(self):
        payload = _payload(chunks=[
            {"content": f"c{i}", "file_path": "legal/x.md"} for i in range(10)
        ])
        assert len(_graph_data_to_chunks(payload, 3)) == 3

    def test_kg_summary_assembled_from_entities_and_relations(self):
        payload = _payload(
            entities=[{"entity_name": "PTO", "entity_type": "policy",
                       "description": "Paid time off policy"}],
            relationships=[{"src_id": "PTO", "tgt_id": "Workday",
                            "description": "PTO is tracked in Workday"}],
        )
        (kg,) = _graph_data_to_chunks(payload, 5)
        assert kg["source_path"] == _GRAPH_KG_SOURCE
        assert "PTO (policy): Paid time off policy" in kg["text"]
        assert "PTO -> Workday: PTO is tracked in Workday" in kg["text"]
        assert kg["chunk_index"] == _GRAPH_CHUNK_INDEX_BASE - 1

    def test_kg_summary_respects_char_cap(self):
        entities = [
            {"entity_name": f"E{i}", "entity_type": "t", "description": "d" * 100}
            for i in range(50)
        ]
        (kg,) = _graph_data_to_chunks(_payload(entities=entities), 5,
                                      kg_summary_max_chars=300)
        assert len(kg["text"]) <= 300 + len(
            "Knowledge-graph facts extracted from the handbook:\n")

    def test_kg_summary_comes_first_when_both_present(self):
        payload = _payload(
            entities=[{"entity_name": "A", "entity_type": "t", "description": "x"}],
            chunks=[{"content": "doc", "file_path": "legal/x.md"}],
        )
        out = _graph_data_to_chunks(payload, 5)
        assert out[0]["source_path"] == _GRAPH_KG_SOURCE
        assert out[1]["source_path"] == "legal/x.md"

    def test_empty_content_chunks_skipped(self):
        payload = _payload(chunks=[{"content": "   ", "file_path": "legal/x.md"}])
        assert _graph_data_to_chunks(payload, 5) == []


class TestGraphStoreReady:
    def test_empty_or_stub_dir_is_not_ready(self, tmp_path):
        assert not _graph_store_ready(str(tmp_path))
        (tmp_path / "kv_store_full_docs.json").write_text("{}")
        assert not _graph_store_ready(str(tmp_path))

    def test_built_store_is_ready(self, tmp_path):
        (tmp_path / "vdb_entities.json").write_text("{}")
        assert _graph_store_ready(str(tmp_path))

    def test_missing_dir_is_not_ready(self, tmp_path):
        assert not _graph_store_ready(str(tmp_path / "nope"))


class TestMaybeGraphTool:
    BASE_CFG = {
        "enabled": True,
        "llm_model": "qwen3.5:4b",
        "embedding_model": "nomic-embed-text",
        "embedding_dim": 768,
        "query_mode": "hybrid",
        "top_k": 20,
        "chunk_top_k": 5,
        "num_ctx": 8192,
    }

    def test_none_when_disabled_or_missing_block(self, tmp_path):
        assert maybe_graph_tool(None, "http://x") is None
        cfg = dict(self.BASE_CFG, enabled=False, working_dir=str(tmp_path))
        assert maybe_graph_tool(cfg, "http://x") is None

    def test_none_when_store_not_built(self, tmp_path):
        cfg = dict(self.BASE_CFG, working_dir=str(tmp_path))
        assert maybe_graph_tool(cfg, "http://x") is None

    def test_tool_built_when_enabled_and_store_ready(self, tmp_path):
        (tmp_path / "vdb_entities.json").write_text("{}")
        cfg = dict(self.BASE_CFG, working_dir=str(tmp_path))
        tool = maybe_graph_tool(cfg, "http://localhost:11434")
        assert tool is not None
        assert tool.query_mode == "hybrid"
        assert tool.chunk_top_k == 5
        assert tool.working_dir == str(tmp_path)
