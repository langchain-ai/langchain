from rl_chain import SelectionScorer
from typing import Dict, Any


class MockScorer(SelectionScorer):
    def score_response(
        self, inputs: Dict[str, Any], llm_response: str, **kwargs
    ) -> float:
        return float(llm_response)


class MockEncoder:
    def encode(self, to_encode):
        return "[encoded]" + to_encode
