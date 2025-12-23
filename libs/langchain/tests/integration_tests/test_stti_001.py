from typing import Any, Callable, Dict, List, Tuple

import pytest


class ProvenanceError(Exception):
    """Raised when a tool argument lacks prior provenance."""


def execute(_: Dict[str, Any]) -> str:
    return "executed"


def gate_factory() -> Tuple[
    Callable[[str, List[str]], None],
    Callable[[Dict[str, Any]], str],
]:
    provenance: Dict[str, List[str]] = {}

    def record(tool_name: str, outputs: List[str]) -> None:
        provenance[tool_name] = [str(o) for o in outputs]

    def gate(tool_call: Dict[str, Any]) -> str:
        for arg in tool_call["args"].values():
            if str(arg) not in {
                v for values in provenance.values() for v in values
            }:
                message = f"Unprovenanced argument: {arg}"
                raise ProvenanceError(message)
        return execute(tool_call)

    return record, gate


def test_stti_001_no_side_effect_without_provenance() -> None:
    record, gate = gate_factory()

    record("list_users", ["bob"])

    assert gate({"name": "delete_user", "args": {"id": "bob"}}) == "executed"

    with pytest.raises(ProvenanceError):
        gate({"name": "delete_user", "args": {"id": "eve"}})
