"""Test pipeline functionality."""
from typing import Dict, List

import pytest
from pydantic import BaseModel

from langchain.chains.base import MultiVariableChain, SingleVariableChain
from langchain.chains.sequential import SequentialChain, SimpleSequentialChain


def fake_call(inputs: Dict[str, str], input_keys: List[str], output_keys: List[str]) -> Dict[str, str]:
    """Fake call function."""
    outputs = {}
    for var in output_keys:
        variables = [inputs[k] for k in input_keys]
        outputs[var] = f"{' '.join(variables)}foo"
    return outputs

class FakeMultiChain(MultiVariableChain, BaseModel):
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
        return fake_call(inputs, self.input_keys, self.output_keys)


class FakeSingleChain(SingleVariableChain, BaseModel):
    """Fake Chain for testing purposes."""

    input_variable: str
    output_variable: str

    @property
    def input_keys(self) -> List[str]:
        """Input keys this chain returns."""
        return [self.input_variable]

    @property
    def output_keys(self) -> List[str]:
        """Input keys this chain returns."""
        return [self.output_variable]

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        return fake_call(inputs, self.input_keys, self.output_keys)


def test_sequential_usage_single_inputs() -> None:
    """Test sequential on single input chains."""
    chain_1 = FakeMultiChain(input_variables=["foo"], output_variables=["bar"])
    chain_2 = FakeMultiChain(input_variables=["bar"], output_variables=["baz"])
    chain = SequentialChain(chains=[chain_1, chain_2], input_variables=["foo"])
    output = chain.run(foo="123")
    expected_output = {"baz": "123foofoo", "foo": "123"}
    assert output == expected_output


def test_sequential_usage_multiple_inputs() -> None:
    """Test sequential on multiple input chains."""
    chain_1 = FakeMultiChain(input_variables=["foo", "test"], output_variables=["bar"])
    chain_2 = FakeMultiChain(input_variables=["bar", "foo"], output_variables=["baz"])
    chain = SequentialChain(chains=[chain_1, chain_2], input_variables=["foo", "test"])
    output = chain({"foo": "123", "test": "456"})
    expected_output = {
        "baz": "123 456foo 123foo",
        "foo": "123",
        "test": "456",
    }
    assert output == expected_output


def test_sequential_usage_multiple_outputs() -> None:
    """Test sequential usage on multiple output chains."""
    chain_1 = FakeMultiChain(input_variables=["foo"], output_variables=["bar", "test"])
    chain_2 = FakeMultiChain(input_variables=["bar", "foo"], output_variables=["baz"])
    chain = SequentialChain(chains=[chain_1, chain_2], input_variables=["foo"])
    output = chain({"foo": "123"})
    expected_output = {
        "baz": "123foo 123foo",
        "foo": "123",
    }
    assert output == expected_output


def test_sequential_missing_inputs() -> None:
    """Test error is raised when input variables are missing."""
    chain_1 = FakeMultiChain(input_variables=["foo"], output_variables=["bar"])
    chain_2 = FakeMultiChain(input_variables=["bar", "test"], output_variables=["baz"])
    with pytest.raises(ValueError):
        # Also needs "test" as an input
        SequentialChain(chains=[chain_1, chain_2], input_variables=["foo"])


def test_sequential_bad_outputs() -> None:
    """Test error is raised when bad outputs are specified."""
    chain_1 = FakeMultiChain(input_variables=["foo"], output_variables=["bar"])
    chain_2 = FakeMultiChain(input_variables=["bar"], output_variables=["baz"])
    with pytest.raises(ValueError):
        # "test" is not present as an output variable.
        SequentialChain(
            chains=[chain_1, chain_2],
            input_variables=["foo"],
            output_variables=["test"],
        )


def test_sequential_valid_outputs() -> None:
    """Test chain runs when valid outputs are specified."""
    chain_1 = FakeMultiChain(input_variables=["foo"], output_variables=["bar"])
    chain_2 = FakeMultiChain(input_variables=["bar"], output_variables=["baz"])
    chain = SequentialChain(
        chains=[chain_1, chain_2],
        input_variables=["foo"],
        output_variables=["bar", "baz"],
    )
    output = chain({"foo": "123"}, return_only_outputs=True)
    expected_output = {"baz": "123foofoo", "bar": "123foo"}
    assert output == expected_output


def test_sequential_overlapping_inputs() -> None:
    """Test error is raised when input variables are overlapping."""
    chain_1 = FakeMultiChain(input_variables=["foo"], output_variables=["bar", "test"])
    chain_2 = FakeMultiChain(input_variables=["bar"], output_variables=["baz"])
    with pytest.raises(ValueError):
        # "test" is specified as an input, but also is an output of one step
        SequentialChain(chains=[chain_1, chain_2], input_variables=["foo", "test"])


def test_simple_sequential_functionality() -> None:
    """Test simple sequential functionality."""
    chain_1 = FakeSingleChain(input_variable="foo", output_variable="bar")
    chain_2 = FakeSingleChain(input_variable="bar", output_variable="baz")
    chain = SimpleSequentialChain(chains=[chain_1, chain_2])
    output = chain.run("123")
    expected_output = "123foofoo"
    assert output == expected_output

    output = chain({"input": "123"})
    expected_output = {"output": "123foofoo", "input": "123"}
    assert output == expected_output
