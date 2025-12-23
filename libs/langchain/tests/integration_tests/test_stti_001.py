import pytest

class ProvenanceError(Exception):
    pass

def execute(call):
    return "executed"

def gate_factory():
    provenance = {}

    def record(tool_name, outputs):
        provenance[tool_name] = [str(o) for o in outputs]

    def gate(tool_call):
        for arg in tool_call["args"].values():
            if str(arg) not in [v for values in provenance.values() for v in values]:
                raise ProvenanceError(f"Unprovenanced argument: {arg}")
        return execute(tool_call)

    return record, gate

def test_stti_001_no_side_effect_without_provenance():
    record, gate = gate_factory()

    # Establish provenance
    record("list_users", ["bob"])

    # PASS: provenanced argument
    assert gate({"name": "delete_user", "args": {"id": "bob"}}) == "executed"

    # FAIL: hallucinated argument
    with pytest.raises(ProvenanceError):
        gate({"name": "delete_user", "args": {"id": "eve"}})
