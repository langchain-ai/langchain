"""Test pipeline functionality."""
from typing import Dict, List, Optional

import pytest

from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.chains.sequential import SequentialChain, SimpleSequentialChain
from langchain.memory.simple import SimpleMemory


class FakeChain(Chain):
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

    def _call(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        outputs = {}
        for var in self.output_variables:
            variables = [inputs[k] for k in self.input_variables]
            outputs[var] = f"{' '.join(variables)}foo"
        return outputs


def test_sequential_usage_single_inputs() -> None:
    """Test sequential on single input chains."""
    chain_1 = FakeChain(input_variables=["foo"], output_variables=["bar"])
    chain_2 = FakeChain(input_variables=["bar"], output_variables=["baz"])
    chain = SequentialChain(chains=[chain_1, chain_2], input_variables=["foo"])
    output = chain({"foo": "123"})
    expected_output = {"baz": "123foofoo", "foo": "123"}
    assert output == expected_output


def test_sequential_usage_multiple_inputs() -> None:
    """Test sequential on multiple input chains."""
    chain_1 = FakeChain(input_variables=["foo", "test"], output_variables=["bar"])
    chain_2 = FakeChain(input_variables=["bar", "foo"], output_variables=["baz"])
    chain = SequentialChain(chains=[chain_1, chain_2], input_variables=["foo", "test"])
    output = chain({"foo": "123", "test": "456"})
    expected_output = {
        "baz": "123 456foo 123foo",
        "foo": "123",
        "test": "456",
    }
    assert output == expected_output


def test_sequential_usage_memory() -> None:
    """Test sequential usage with memory."""
    memory = SimpleMemory(memories={"zab": "rab"})
    chain_1 = FakeChain(input_variables=["foo"], output_variables=["bar"])
    chain_2 = FakeChain(input_variables=["bar"], output_variables=["baz"])
    chain = SequentialChain(
        memory=memory, chains=[chain_1, chain_2], input_variables=["foo"]
    )
    output = chain({"foo": "123"})
    expected_output = {"baz": "123foofoo", "foo": "123", "zab": "rab"}
    assert output == expected_output
    memory = SimpleMemory(memories={"zab": "rab", "foo": "rab"})
    chain_1 = FakeChain(input_variables=["foo"], output_variables=["bar"])
    chain_2 = FakeChain(input_variables=["bar"], output_variables=["baz"])
    with pytest.raises(ValueError):
        SequentialChain(
            memory=memory, chains=[chain_1, chain_2], input_variables=["foo"]
        )


def test_sequential_usage_multiple_outputs() -> None:
    """Test sequential usage on multiple output chains."""
    chain_1 = FakeChain(input_variables=["foo"], output_variables=["bar", "test"])
    chain_2 = FakeChain(input_variables=["bar", "foo"], output_variables=["baz"])
    chain = SequentialChain(chains=[chain_1, chain_2], input_variables=["foo"])
    output = chain({"foo": "123"})
    expected_output = {
        "baz": "123foo 123foo",
        "foo": "123",
    }
    assert output == expected_output


def test_sequential_missing_inputs() -> None:
    """Test error is raised when input variables are missing."""
    chain_1 = FakeChain(input_variables=["foo"], output_variables=["bar"])
    chain_2 = FakeChain(input_variables=["bar", "test"], output_variables=["baz"])
    with pytest.raises(ValueError):
        # Also needs "test" as an input
        SequentialChain(chains=[chain_1, chain_2], input_variables=["foo"])


def test_sequential_bad_outputs() -> None:
    """Test error is raised when bad outputs are specified."""
    chain_1 = FakeChain(input_variables=["foo"], output_variables=["bar"])
    chain_2 = FakeChain(input_variables=["bar"], output_variables=["baz"])
    with pytest.raises(ValueError):
        # "test" is not present as an output variable.
        SequentialChain(
            chains=[chain_1, chain_2],
            input_variables=["foo"],
            output_variables=["test"],
        )


def test_sequential_valid_outputs() -> None:
    """Test chain runs when valid outputs are specified."""
    chain_1 = FakeChain(input_variables=["foo"], output_variables=["bar"])
    chain_2 = FakeChain(input_variables=["bar"], output_variables=["baz"])
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
    chain_1 = FakeChain(input_variables=["foo"], output_variables=["bar", "test"])
    chain_2 = FakeChain(input_variables=["bar"], output_variables=["baz"])
    with pytest.raises(ValueError):
        # "test" is specified as an input, but also is an output of one step
        SequentialChain(chains=[chain_1, chain_2], input_variables=["foo", "test"])


def test_simple_sequential_functionality() -> None:
    """Test simple sequential functionality."""
    chain_1 = FakeChain(input_variables=["foo"], output_variables=["bar"])
    chain_2 = FakeChain(input_variables=["bar"], output_variables=["baz"])
    chain = SimpleSequentialChain(chains=[chain_1, chain_2])
    output = chain({"input": "123"})
    expected_output = {"output": "123foofoo", "input": "123"}
    assert output == expected_output


def test_multi_input_errors() -> None:
    """Test simple sequential errors if multiple input variables are expected."""
    chain_1 = FakeChain(input_variables=["foo"], output_variables=["bar"])
    chain_2 = FakeChain(input_variables=["bar", "foo"], output_variables=["baz"])
    with pytest.raises(ValueError):
        SimpleSequentialChain(chains=[chain_1, chain_2])


def test_multi_output_errors() -> None:
    """Test simple sequential errors if multiple output variables are expected."""
    chain_1 = FakeChain(input_variables=["foo"], output_variables=["bar", "grok"])
    chain_2 = FakeChain(input_variables=["bar"], output_variables=["baz"])
    with pytest.raises(ValueError):
        SimpleSequentialChain(chains=[chain_1, chain_2])
