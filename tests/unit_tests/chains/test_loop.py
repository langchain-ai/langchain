"""Test pipeline functionality."""
from typing import Any, Dict, List

import pytest
from pydantic import BaseModel

from langchain.chains.base import Chain
from langchain.chains.loop import BaseWhileChain


class NumWordsChain(Chain, BaseModel):
    """Dummy Chain for testing purposes. Returns a string with num_words space separated 'a's"""

    input_key: str = "num_words"
    output_key: str = "text"

    @property
    def input_keys(self) -> List[str]:
        """Input keys this chain returns."""
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Input keys this chain returns."""
        return [self.output_key]

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        outputs = {}
        required_num_of_words = inputs[self.input_key]
        outputs[self.output_key] = " ".join(["a"] * required_num_of_words)
        return outputs


class WhileWordsChain(BaseWhileChain):
    """Implementation of BaseWhileChain that terminates when a sentence with 10 words is generated"""

    def get_initial_state(self, inputs: Dict[str, str]) -> Any:
        state = {"num_words": inputs["initial_state"]}
        return state

    def get_updated_state(self, current_state: Any, inputs: Dict[str, str]) -> Any:
        updated_state = {"num_words": current_state["num_words"] + inputs["increment"]}
        return updated_state

    def stopping_criterion(self, outputs: Dict[str, str]) -> bool:
        has_10_words = len(outputs["text"].split()) == 10
        return has_10_words


def test_while_chain() -> None:
    """Test an implementation of BaseWhileChain using a fake chain that returns a sentence with num_words words"""

    num_words_chain = NumWordsChain()
    while_words_chain = WhileWordsChain(
        chain=num_words_chain, input_variables=["initial_state", "increment"]
    )
    output = while_words_chain(
        inputs={
            "initial_state": 0,
            "increment": 1,
        }
    )
    expected_output = {
        "initial_state": 0,
        "increment": 1,
        "output": {"num_words": 10, "text": "a a a a a a a a a a"},
    }
    assert output == expected_output
