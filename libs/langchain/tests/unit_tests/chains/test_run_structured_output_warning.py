import warnings
import pytest
from langchain_classic.chains.base import Chain
from typing import Dict, List, Any

class DummyStructuredChain(Chain):
    """Minimal Chain that returns a dict to emulate structured output chain."""
    
    @property
    def input_keys(self) -> List[str]:
        return ["input"]
        
    @property
    def output_keys(self) -> List[str]:
        return ["output"]

    def _call(self, inputs: Dict[str, Any], run_manager=None) -> Dict[str, Any]:
        # return a structured dict (simulate structured output chain)
        # The 'output' key contains the structured result (a dict)
        return {"output": {"field1": "value1", "field2": 2}}

def test_run_emits_warning_for_structured_output() -> None:
    c = DummyStructuredChain()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = c.run("ignore this input")
        
        # assert we got a dict back 
        assert isinstance(result, dict)
        assert result == {"field1": "value1", "field2": 2}
        
        # assert warning was emitted
        assert len(w) > 0
        assert any("structured output" in str(warn.message).lower() for warn in w)
