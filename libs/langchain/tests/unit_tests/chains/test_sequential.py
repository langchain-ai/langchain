"""Test pipeline functionality."""

import re

import pytest
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from typing_extensions import override

from langchain_classic.chains.base import Chain
from langchain_classic.chains.sequential import SequentialChain, SimpleSequentialChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.memory.simple import SimpleMemory
from tests.unit_tests.callbacks.fake_callback_handler import FakeCallbackHandler


class FakeChain(Chain):
    """Fake Chain for testing purposes."""

    input_variables: list[str]
    output_variables: list[str]

    @property
    def input_keys(self) -> list[str]:
        """Input keys this chain returns."""
        return self.input_variables

    @property
    def output_keys(self) -> list[str]:
        """Input keys this chain returns."""
        return self.output_variables

    @override
    def _call(
        self,
        inputs: dict[str, str],
        run_manager: CallbackManagerForChainRun | None = None,
    ) -> dict[str, str]:
        outputs = {}
        for var in self.output_variables:
            variables = [inputs[k] for k in self.input_variables]
            outputs[var] = f"{' '.join(variables)}foo"
        return outputs

    @override
    async def _acall(
        self,
        inputs: dict[str, str],
        run_manager: AsyncCallbackManagerForChainRun | None = None,
    ) -> dict[str, str]:
        outputs = {}
        for var in self.output_variables:
            variables = [inputs[k] for k in self.input_variables]
            outputs[var] = f"{' '.join(variables)}foo"
        return outputs


def test_sequential_usage_single_inputs() -> None:
    """Test sequential on single input chains."""
    chain_1 = FakeChain(input_variables=["foo"], output_variables=["bar"])
    chain_2 = FakeChain(input_variables=["bar"], output_variables=["baz"])
    chain = SequentialChain(chains=[chain_1, chain_2], input_variables=["foo"])  # type: ignore[call-arg]
    output = chain({"foo": "123"})
    expected_output = {"baz": "123foofoo", "foo": "123"}
    assert output == expected_output


def test_sequential_usage_multiple_inputs() -> None:
    """Test sequential on multiple input chains."""
    chain_1 = FakeChain(input_variables=["foo", "test"], output_variables=["bar"])
    chain_2 = FakeChain(input_variables=["bar", "foo"], output_variables=["baz"])
    chain = SequentialChain(chains=[chain_1, chain_2], input_variables=["foo", "test"])  # type: ignore[call-arg]
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
    chain = SequentialChain(  # type: ignore[call-arg]
        memory=memory,
        chains=[chain_1, chain_2],
        input_variables=["foo"],
    )
    output = chain({"foo": "123"})
    expected_output = {"baz": "123foofoo", "foo": "123", "zab": "rab"}
    assert output == expected_output
    memory = SimpleMemory(memories={"zab": "rab", "foo": "rab"})
    chain_1 = FakeChain(input_variables=["foo"], output_variables=["bar"])
    chain_2 = FakeChain(input_variables=["bar"], output_variables=["baz"])
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Value error, The input key(s) foo are found in the Memory keys"
        ),
    ):
        SequentialChain(  # type: ignore[call-arg]
            memory=memory,
            chains=[chain_1, chain_2],
            input_variables=["foo"],
        )


def test_sequential_internal_chain_use_memory() -> None:
    """Test sequential usage with memory for one of the internal chains."""
    memory = ConversationBufferMemory(memory_key="bla")
    memory.save_context({"input": "yo"}, {"output": "ya"})
    chain_1 = FakeChain(
        input_variables=["foo", "bla"],
        output_variables=["bar"],
        memory=memory,
    )
    chain_2 = FakeChain(input_variables=["bar"], output_variables=["baz"])
    chain = SequentialChain(chains=[chain_1, chain_2], input_variables=["foo"])  # type: ignore[call-arg]
    output = chain({"foo": "123"})
    print("HEYYY OUTPUT", output)  # noqa: T201
    expected_output = {"foo": "123", "baz": "123 Human: yo\nAI: yafoofoo"}
    assert output == expected_output


def test_sequential_usage_multiple_outputs() -> None:
    """Test sequential usage on multiple output chains."""
    chain_1 = FakeChain(input_variables=["foo"], output_variables=["bar", "test"])
    chain_2 = FakeChain(input_variables=["bar", "foo"], output_variables=["baz"])
    chain = SequentialChain(chains=[chain_1, chain_2], input_variables=["foo"])  # type: ignore[call-arg]
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
    with pytest.raises(
        ValueError,
        match=re.escape("Value error, Missing required input keys: {'test'}"),
    ):
        # Also needs "test" as an input
        SequentialChain(chains=[chain_1, chain_2], input_variables=["foo"])  # type: ignore[call-arg]


def test_sequential_bad_outputs() -> None:
    """Test error is raised when bad outputs are specified."""
    chain_1 = FakeChain(input_variables=["foo"], output_variables=["bar"])
    chain_2 = FakeChain(input_variables=["bar"], output_variables=["baz"])
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Value error, Expected output variables that were not found: {'test'}."
        ),
    ):
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
    with pytest.raises(
        ValueError, match="Value error, Chain returned keys that already exist"
    ):
        # "test" is specified as an input, but also is an output of one step
        SequentialChain(chains=[chain_1, chain_2], input_variables=["foo", "test"])  # type: ignore[call-arg]


def test_simple_sequential_functionality() -> None:
    """Test simple sequential functionality."""
    chain_1 = FakeChain(input_variables=["foo"], output_variables=["bar"])
    chain_2 = FakeChain(input_variables=["bar"], output_variables=["baz"])
    chain = SimpleSequentialChain(chains=[chain_1, chain_2])
    output = chain({"input": "123"})
    expected_output = {"output": "123foofoo", "input": "123"}
    assert output == expected_output


@pytest.mark.parametrize("is_async", [False, True])
async def test_simple_sequential_functionality_with_callbacks(
    *, is_async: bool
) -> None:
    """Test simple sequential functionality."""
    handler_1 = FakeCallbackHandler()
    handler_2 = FakeCallbackHandler()
    handler_3 = FakeCallbackHandler()
    chain_1 = FakeChain(
        input_variables=["foo"],
        output_variables=["bar"],
        callbacks=[handler_1],
    )
    chain_2 = FakeChain(
        input_variables=["bar"],
        output_variables=["baz"],
        callbacks=[handler_2],
    )
    chain_3 = FakeChain(
        input_variables=["jack"],
        output_variables=["baf"],
        callbacks=[handler_3],
    )
    chain = SimpleSequentialChain(chains=[chain_1, chain_2, chain_3])
    if is_async:
        output = await chain.ainvoke({"input": "123"})
    else:
        output = chain({"input": "123"})
    expected_output = {"output": "123foofoofoo", "input": "123"}
    assert output == expected_output
    # Check that each of the callbacks were invoked once per the entire run
    for handler in [handler_1, handler_2, handler_3]:
        assert handler.starts == 1
        assert handler.ends == 1
        assert handler.errors == 0


def test_multi_input_errors() -> None:
    """Test simple sequential errors if multiple input variables are expected."""
    chain_1 = FakeChain(input_variables=["foo"], output_variables=["bar"])
    chain_2 = FakeChain(input_variables=["bar", "foo"], output_variables=["baz"])
    with pytest.raises(
        ValueError,
        match="Value error, Chains used in SimplePipeline should all have one input",
    ):
        SimpleSequentialChain(chains=[chain_1, chain_2])


def test_multi_output_errors() -> None:
    """Test simple sequential errors if multiple output variables are expected."""
    chain_1 = FakeChain(input_variables=["foo"], output_variables=["bar", "grok"])
    chain_2 = FakeChain(input_variables=["bar"], output_variables=["baz"])
    with pytest.raises(
        ValueError,
        match="Value error, Chains used in SimplePipeline should all have one output",
    ):
        SimpleSequentialChain(chains=[chain_1, chain_2])
