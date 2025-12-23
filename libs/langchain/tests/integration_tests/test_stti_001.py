from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

import pytest


class ProvenanceError(Exception):
    """Raised when a tool argument lacks prior provenance."""


def execute(call: Dict[str, Any]) -> str:
    """Stub side-effect execution."""
    return "executed"


def gate_factory() -> Tuple[
    Callable[[str, List[str]], None],
    Callable[[Dict[str, Any]], str],
]:
    provenance: Dict[str, List[str]] = {}

    def record(tool_name: str, outputs: List[str]) -> None:
        provenance[tool_name] = [str(o) for o in outputs]

    def gate(tool_call: Dict[str, Any]) -> str:
        values: List[str] = [
            v for outputs in provenance.values() for v in outputs
        ]
        for arg in tool_call["args"].values():
            if str(arg) not in values:
                message = f"Unprovenanced argument: {arg}"
                raise ProvenanceError(message)
        return execute(tool_call)

    return record, gate


@pytest.mark.integration
def test_stti_001_no_side_effect_without_provenance() -> None:
    record, gate = gate_factory()

    # Establish provenance
    record("list_users", ["bob"])

    # PASS: argument originates from prior tool output
    result = gate({"name": "delete_user", "args": {"id": "bob"}})
    assert result == "executed"

    # FAIL: hallucinated argument with no provenance
    with pytest.raises(ProvenanceError):
        gate({"name": "delete_user", "args": {"id": "eve"}})
