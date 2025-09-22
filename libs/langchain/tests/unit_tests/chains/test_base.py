"""Test logic on base chain class."""

import uuid
from typing import Any, Optional

import pytest
from langchain_core.callbacks.manager import CallbackManagerForChainRun
from langchain_core.memory import BaseMemory
from langchain_core.tracers.context import collect_runs
from typing_extensions import override

from langchain.chains.base import Chain
from langchain.schema import RUN_KEY
from tests.unit_tests.callbacks.fake_callback_handler import FakeCallbackHandler


class FakeMemory(BaseMemory):
    """Fake memory class for testing purposes."""

    @property
    def memory_variables(self) -> list[str]:
        """Return baz variable."""
        return ["baz"]

    @override
    def load_memory_variables(
        self,
        inputs: Optional[dict[str, Any]] = None,
    ) -> dict[str, str]:
        """Return baz variable."""
        return {"baz": "foo"}

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, str]) -> None:
        """Pass."""

    def clear(self) -> None:
        """Pass."""


class FakeChain(Chain):
    """Fake chain class for testing purposes."""

    be_correct: bool = True
    the_input_keys: list[str] = ["foo"]
    the_output_keys: list[str] = ["bar"]

    @property
    def input_keys(self) -> list[str]:
        """Input keys."""
        return self.the_input_keys

    @property
    def output_keys(self) -> list[str]:
        """Output key of bar."""
        return self.the_output_keys

    @override
    def _call(
        self,
        inputs: dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> dict[str, str]:
        if self.be_correct:
            return {"bar": "baz"}
        return {"baz": "bar"}


def test_bad_inputs() -> None:
    """Test errors are raised if input keys are not found."""
    chain = FakeChain()
    with pytest.raises(ValueError, match="Missing some input keys: {'foo'}"):
        chain({"foobar": "baz"})


def test_bad_outputs() -> None:
    """Test errors are raised if outputs keys are not found."""
    chain = FakeChain(be_correct=False)
    with pytest.raises(ValueError, match="Missing some output keys: {'bar'}"):
        chain({"foo": "baz"})


def test_run_info() -> None:
    """Test that run_info is returned properly when specified."""
    chain = FakeChain()
    output = chain({"foo": "bar"}, include_run_info=True)
    assert "foo" in output
    assert "bar" in output
    assert RUN_KEY in output


def test_correct_call() -> None:
    """Test correct call of fake chain."""
    chain = FakeChain()
    output = chain({"foo": "bar"})
    assert output == {"foo": "bar", "bar": "baz"}


def test_single_input_correct() -> None:
    """Test passing single input works."""
    chain = FakeChain()
    output = chain("bar")
    assert output == {"foo": "bar", "bar": "baz"}


def test_single_input_error() -> None:
    """Test passing single input errors as expected."""
    chain = FakeChain(the_input_keys=["foo", "bar"])
    with pytest.raises(ValueError, match="Missing some input keys:"):
        chain("bar")


def test_run_single_arg() -> None:
    """Test run method with single arg."""
    chain = FakeChain()
    output = chain.run("bar")
    assert output == "baz"


def test_run_multiple_args_error() -> None:
    """Test run method with multiple args errors as expected."""
    chain = FakeChain()
    with pytest.raises(
        ValueError, match="`run` supports only one positional argument."
    ):
        chain.run("bar", "foo")


def test_run_kwargs() -> None:
    """Test run method with kwargs."""
    chain = FakeChain(the_input_keys=["foo", "bar"])
    output = chain.run(foo="bar", bar="foo")
    assert output == "baz"


def test_run_kwargs_error() -> None:
    """Test run method with kwargs errors as expected."""
    chain = FakeChain(the_input_keys=["foo", "bar"])
    with pytest.raises(ValueError, match="Missing some input keys: {'bar'}"):
        chain.run(foo="bar", baz="foo")


def test_run_args_and_kwargs_error() -> None:
    """Test run method with args and kwargs."""
    chain = FakeChain(the_input_keys=["foo", "bar"])
    with pytest.raises(
        ValueError,
        match="`run` supported with either positional arguments "
        "or keyword arguments but not both.",
    ):
        chain.run("bar", foo="bar")


def test_multiple_output_keys_error() -> None:
    """Test run with multiple output keys errors as expected."""
    chain = FakeChain(the_output_keys=["foo", "bar"])
    with pytest.raises(
        ValueError,
        match="`run` not supported when there is not exactly one output key.",
    ):
        chain.run("bar")


def test_run_arg_with_memory() -> None:
    """Test run method works when arg is passed."""
    chain = FakeChain(the_input_keys=["foo", "baz"], memory=FakeMemory())
    chain.run("bar")


def test_run_with_callback() -> None:
    """Test run method works when callback manager is passed."""
    handler = FakeCallbackHandler()
    chain = FakeChain(
        callbacks=[handler],
    )
    output = chain.run("bar")
    assert output == "baz"
    assert handler.starts == 1
    assert handler.ends == 1
    assert handler.errors == 0


def test_run_with_callback_and_input_error() -> None:
    """Test callback manager catches run validation input error."""
    handler = FakeCallbackHandler()
    chain = FakeChain(
        the_input_keys=["foo", "bar"],
        callbacks=[handler],
    )

    with pytest.raises(ValueError, match="Missing some input keys: {'foo'}"):
        chain({"bar": "foo"})

    assert handler.starts == 1
    assert handler.ends == 0
    assert handler.errors == 1


def test_manually_specify_rid() -> None:
    chain = FakeChain()
    run_id = uuid.uuid4()
    with collect_runs() as cb:
        chain.invoke({"foo": "bar"}, {"run_id": run_id})
        run = cb.traced_runs[0]
        assert run.id == run_id

    run_id2 = uuid.uuid4()
    with collect_runs() as cb:
        list(chain.stream({"foo": "bar"}, {"run_id": run_id2}))
        run = cb.traced_runs[0]
        assert run.id == run_id2


async def test_manually_specify_rid_async() -> None:
    chain = FakeChain()
    run_id = uuid.uuid4()
    with collect_runs() as cb:
        await chain.ainvoke({"foo": "bar"}, {"run_id": run_id})
        run = cb.traced_runs[0]
        assert run.id == run_id
    run_id2 = uuid.uuid4()
    with collect_runs() as cb:
        res = chain.astream({"foo": "bar"}, {"run_id": run_id2})
        async for _ in res:
            pass
        run = cb.traced_runs[0]
        assert run.id == run_id2


def test_run_with_callback_and_output_error() -> None:
    """Test callback manager catches run validation output error."""
    handler = FakeCallbackHandler()
    chain = FakeChain(
        the_output_keys=["foo", "bar"],
        callbacks=[handler],
    )

    with pytest.raises(ValueError, match="Missing some output keys: {'foo'}"):
        chain("foo")

    assert handler.starts == 1
    assert handler.ends == 0
    assert handler.errors == 1
