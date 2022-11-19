"""Test functionality around the simple pipeline chain."""

from typing import Dict, List

import pytest
from pydantic import BaseModel

from langchain.chains.base import Chain
from langchain.chains.simple_pipeline import SimplePipeline


class FakeChain(Chain, BaseModel):
    """Fake chain for testing purposes."""

    input_variables: List[str]
    output_variables: List[str]

    @property
    def input_keys(self) -> List[str]:
        """Input keys this chain returns."""
        return self.input_variables

    @property
    def output_keys(self) -> List[str]:
        """Input keys this chain returns."""
        return self.output_variables

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        outputs = {}
        for var in self.output_variables:
            variables = [inputs[k] for k in self.input_variables]
            outputs[var] = " ".join(variables) + "foo"
        return outputs


def test_pipeline_functionality() -> None:
    """Test simple pipeline functionality."""
    chain_1 = FakeChain(input_variables=["foo"], output_variables=["bar"])
    chain_2 = FakeChain(input_variables=["bar"], output_variables=["baz"])
    pipeline = SimplePipeline(chains=[chain_1, chain_2])
    output = pipeline({"input": "123"})
    expected_output = {"output": "123foofoo", "input": "123"}
    assert output == expected_output


def test_multi_input_errors() -> None:
    """Test pipeline errors if multiple input variables are expected."""
    chain_1 = FakeChain(input_variables=["foo"], output_variables=["bar"])
    chain_2 = FakeChain(input_variables=["bar", "foo"], output_variables=["baz"])
    with pytest.raises(ValueError):
        SimplePipeline(chains=[chain_1, chain_2])


def test_multi_output_errors() -> None:
    """Test pipeline errors if multiple output variables are expected."""
    chain_1 = FakeChain(input_variables=["foo"], output_variables=["bar", "grok"])
    chain_2 = FakeChain(input_variables=["bar"], output_variables=["baz"])
    with pytest.raises(ValueError):
        SimplePipeline(chains=[chain_1, chain_2])
