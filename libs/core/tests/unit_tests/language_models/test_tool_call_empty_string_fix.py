"""Tests for GH-39335: empty string tool-call delta should not overwrite id/name.

When streaming tool calls, some providers send deltas with empty string
id/name ("") after the initial chunk that carried the real values.
These empty strings must not clobber previously accumulated state.
"""

from __future__ import annotations

import pytest

from langchain_core.language_models.chat_model_stream import (
    _merge_block_delta_into_store,
    _merge_chunk_into_store,
)
from langchain_core.language_models._compat_bridge import _accumulate


# ---------------------------------------------------------------------------
# _merge_block_delta_into_store (chat_model_stream.py)
# ---------------------------------------------------------------------------


class TestMergeBlockDeltaIntoStoreEmptyString:
    """Empty string id/name must not overwrite existing values."""

    def test_empty_string_id_does_not_overwrite(self) -> None:
        store: dict[int, dict] = {}
        # First delta: real id
        _merge_block_delta_into_store(store, 0, {"id": "tc_1", "name": "my_tool", "args": ""})
        assert store[0]["id"] == "tc_1"
        # Second delta: empty string id should NOT overwrite
        _merge_block_delta_into_store(store, 0, {"id": "", "name": "", "args": '{"x":'})
        assert store[0]["id"] == "tc_1"
        assert store[0]["name"] == "my_tool"

    def test_empty_string_name_does_not_overwrite(self) -> None:
        store: dict[int, dict] = {}
        _merge_block_delta_into_store(store, 0, {"id": "tc_2", "name": "search", "args": ""})
        _merge_block_delta_into_store(store, 0, {"id": "", "name": "", "args": '{"q":'})
        assert store[0]["name"] == "search"

    def test_none_still_does_not_overwrite(self) -> None:
        store: dict[int, dict] = {}
        _merge_block_delta_into_store(store, 0, {"id": "tc_3", "name": "calc", "args": ""})
        _merge_block_delta_into_store(store, 0, {"id": None, "name": None, "args": "1"})
        assert store[0]["id"] == "tc_3"
        assert store[0]["name"] == "calc"

    def test_non_empty_values_still_merge(self) -> None:
        """Non-empty id/name on first delta should be stored normally."""
        store: dict[int, dict] = {}
        _merge_block_delta_into_store(store, 0, {"id": "tc_4", "name": "fetch", "args": ""})
        assert store[0]["id"] == "tc_4"
        assert store[0]["name"] == "fetch"

    def test_args_still_concatenated(self) -> None:
        """Args should still be concatenated even when id/name are empty."""
        store: dict[int, dict] = {}
        _merge_block_delta_into_store(store, 0, {"id": "tc_5", "name": "run", "args": '{"a"'})
        _merge_block_delta_into_store(store, 0, {"id": "", "name": "", "args": ': 1}'})
        assert store[0]["args"] == '{"a": 1}'
        assert store[0]["id"] == "tc_5"
        assert store[0]["name"] == "run"


# ---------------------------------------------------------------------------
# _merge_chunk_into_store (chat_model_stream.py)
# ---------------------------------------------------------------------------


class TestMergeChunkIntoStoreEmptyString:
    """The legacy merge path should also be safe (it already uses .get() truthiness)."""

    def test_empty_string_id_does_not_overwrite(self) -> None:
        store: dict[int, dict] = {}
        _merge_chunk_into_store(store, 0, {"id": "tc_1", "name": "my_tool", "args": ""})
        _merge_chunk_into_store(store, 0, {"id": "", "name": "", "args": '{"x":'})
        assert store[0]["id"] == "tc_1"
        assert store[0]["name"] == "my_tool"


# ---------------------------------------------------------------------------
# _accumulate (_compat_bridge.py)
# ---------------------------------------------------------------------------


class TestAccumulateEmptyString:
    """_accumulate should not let empty string id/name clobber state."""

    def test_empty_string_id_does_not_overwrite(self) -> None:
        state = {"type": "tool_call_chunk", "id": "tc_1", "name": "search", "args": ""}
        delta = {"type": "tool_call_chunk", "id": "", "name": "", "args": '{"q":'}
        result = _accumulate(state, delta)
        assert result["id"] == "tc_1"
        assert result["name"] == "search"
        assert result["args"] == '{"q":'

    def test_empty_string_name_does_not_overwrite(self) -> None:
        state = {"type": "tool_call_chunk", "id": "tc_2", "name": "calc", "args": ""}
        delta = {"type": "tool_call_chunk", "id": "", "name": "", "args": "1+1"}
        result = _accumulate(state, delta)
        assert result["name"] == "calc"

    def test_server_tool_call_chunk_same_behavior(self) -> None:
        state = {"type": "server_tool_call_chunk", "id": "stc_1", "name": "exec", "args": ""}
        delta = {"type": "server_tool_call_chunk", "id": "", "name": "", "args": "code"}
        result = _accumulate(state, delta)
        assert result["id"] == "stc_1"
        assert result["name"] == "exec"
        assert result["args"] == "code"

    def test_non_empty_id_still_overwrites(self) -> None:
        """A real (non-empty) id on a later delta should still be picked up."""
        state = {"type": "tool_call_chunk", "id": None, "name": None, "args": ""}
        delta = {"type": "tool_call_chunk", "id": "tc_late", "name": "late_tool", "args": ""}
        result = _accumulate(state, delta)
        assert result["id"] == "tc_late"
        assert result["name"] == "late_tool"

    def test_none_id_does_not_overwrite(self) -> None:
        state = {"type": "tool_call_chunk", "id": "tc_3", "name": "tool", "args": ""}
        delta = {"type": "tool_call_chunk", "id": None, "name": None, "args": "x"}
        result = _accumulate(state, delta)
        assert result["id"] == "tc_3"
        assert result["name"] == "tool"

    def test_args_still_concatenated(self) -> None:
        state = {"type": "tool_call_chunk", "id": "tc_4", "name": "run", "args": '{"a"'}
        delta = {"type": "tool_call_chunk", "id": "", "name": "", "args": ': 1}'}
        result = _accumulate(state, delta)
        assert result["args"] == '{"a": 1}'
        assert result["id"] == "tc_4"
