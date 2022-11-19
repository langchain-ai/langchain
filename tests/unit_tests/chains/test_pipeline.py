"""Test pipeline functionality."""
from typing import Dict, List

import pytest
from pydantic import BaseModel

from langchain.chains.base import Chain
from langchain.chains.pipeline import Pipeline


class FakeChain(Chain, BaseModel):
    """Fake Chain for testing purposes."""

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


def test_pipeline_usage_single_inputs() -> None:
    """Test pipeline on single input chains."""
    chain_1 = FakeChain(input_variables=["foo"], output_variables=["bar"])
    chain_2 = FakeChain(input_variables=["bar"], output_variables=["baz"])
    pipeline = Pipeline(chains=[chain_1, chain_2], input_variables=["foo"])
    output = pipeline({"foo": "123"})
    expected_output = {"bar": "123foo", "baz": "123foofoo", "foo": "123"}
    assert output == expected_output


def test_pipeline_usage_multiple_inputs() -> None:
    """Test pipeline on multiple input chains."""
    chain_1 = FakeChain(input_variables=["foo", "test"], output_variables=["bar"])
    chain_2 = FakeChain(input_variables=["bar", "foo"], output_variables=["baz"])
    pipeline = Pipeline(chains=[chain_1, chain_2], input_variables=["foo", "test"])
    output = pipeline({"foo": "123", "test": "456"})
    expected_output = {
        "bar": "123 456foo",
        "baz": "123 456foo 123foo",
        "foo": "123",
        "test": "456",
    }
    assert output == expected_output


def test_pipeline_usage_multiple_outputs() -> None:
    """Test pipeline usage on multiple output chains."""
    chain_1 = FakeChain(input_variables=["foo"], output_variables=["bar", "test"])
    chain_2 = FakeChain(input_variables=["bar", "foo"], output_variables=["baz"])
    pipeline = Pipeline(chains=[chain_1, chain_2], input_variables=["foo"])
    output = pipeline({"foo": "123"})
    expected_output = {
        "bar": "123foo",
        "baz": "123foo 123foo",
        "foo": "123",
        "test": "123foo",
    }
    assert output == expected_output


def test_pipeline_missing_inputs() -> None:
    """Test error is raised when input variables are missing."""
    chain_1 = FakeChain(input_variables=["foo"], output_variables=["bar"])
    chain_2 = FakeChain(input_variables=["bar", "test"], output_variables=["baz"])
    with pytest.raises(ValueError):
        # Also needs "test" as an input
        Pipeline(chains=[chain_1, chain_2], input_variables=["foo"])


def test_pipeline_bad_outputs() -> None:
    """Test error is raised when bad outputs are specified."""
    chain_1 = FakeChain(input_variables=["foo"], output_variables=["bar"])
    chain_2 = FakeChain(input_variables=["bar"], output_variables=["baz"])
    with pytest.raises(ValueError):
        # "test" is not present as an output variable.
        Pipeline(
            chains=[chain_1, chain_2],
            input_variables=["foo"],
            output_variables=["test"],
        )


def test_pipeline_overlapping_inputs() -> None:
    """Test error is raised when input variables are overlapping."""
    chain_1 = FakeChain(input_variables=["foo"], output_variables=["bar", "test"])
    chain_2 = FakeChain(input_variables=["bar"], output_variables=["baz"])
    with pytest.raises(ValueError):
        # "test" is specified as an input, but also is an output of one step
        Pipeline(chains=[chain_1, chain_2], input_variables=["foo", "test"])
