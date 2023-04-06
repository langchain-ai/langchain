"""Test logic on base chain class."""
from typing import Any, Dict, List, Optional

import pytest

from langchain.callbacks.base import CallbackManager
from langchain.chains.base import Chain
from langchain.schema import BaseMemory
from tests.unit_tests.callbacks.fake_callback_handler import FakeCallbackHandler


class FakeMemory(BaseMemory):
    """Fake memory class for testing purposes."""

    @property
    def memory_variables(self) -> List[str]:
        """Return baz variable."""
        return ["baz"]

    def load_memory_variables(
        self, inputs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """Return baz variable."""
        return {"baz": "foo"}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Pass."""
        pass

    def clear(self) -> None:
        """Pass."""
        pass


class FakeChain(Chain):
    """Fake chain class for testing purposes."""

    be_correct: bool = True
    the_input_keys: List[str] = ["foo"]
    the_output_keys: List[str] = ["bar"]

    @property
    def input_keys(self) -> List[str]:
        """Input keys."""
        return self.the_input_keys

    @property
    def output_keys(self) -> List[str]:
        """Output key of bar."""
        return self.the_output_keys

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        if self.be_correct:
            return {"bar": "baz"}
        else:
            return {"baz": "bar"}


def test_bad_inputs() -> None:
    """Test errors are raised if input keys are not found."""
    chain = FakeChain()
    with pytest.raises(ValueError):
        chain({"foobar": "baz"})


def test_bad_outputs() -> None:
    """Test errors are raised if outputs keys are not found."""
    chain = FakeChain(be_correct=False)
    with pytest.raises(ValueError):
        chain({"foo": "baz"})


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
    with pytest.raises(ValueError):
        chain("bar")


def test_run_single_arg() -> None:
    """Test run method with single arg."""
    chain = FakeChain()
    output = chain.run("bar")
    assert output == "baz"


def test_run_multiple_args_error() -> None:
    """Test run method with multiple args errors as expected."""
    chain = FakeChain()
    with pytest.raises(ValueError):
        chain.run("bar", "foo")


def test_run_kwargs() -> None:
    """Test run method with kwargs."""
    chain = FakeChain(the_input_keys=["foo", "bar"])
    output = chain.run(foo="bar", bar="foo")
    assert output == "baz"


def test_run_kwargs_error() -> None:
    """Test run method with kwargs errors as expected."""
    chain = FakeChain(the_input_keys=["foo", "bar"])
    with pytest.raises(ValueError):
        chain.run(foo="bar", baz="foo")


def test_run_args_and_kwargs_error() -> None:
    """Test run method with args and kwargs."""
    chain = FakeChain(the_input_keys=["foo", "bar"])
    with pytest.raises(ValueError):
        chain.run("bar", foo="bar")


def test_multiple_output_keys_error() -> None:
    """Test run with multiple output keys errors as expected."""
    chain = FakeChain(the_output_keys=["foo", "bar"])
    with pytest.raises(ValueError):
        chain.run("bar")


def test_run_arg_with_memory() -> None:
    """Test run method works when arg is passed."""
    chain = FakeChain(the_input_keys=["foo", "baz"], memory=FakeMemory())
    chain.run("bar")


def test_run_with_callback() -> None:
    """Test run method works when callback manager is passed."""
    handler = FakeCallbackHandler()
    chain = FakeChain(
        callback_manager=CallbackManager(handlers=[handler]), verbose=True
    )
    output = chain.run("bar")
    assert output == "baz"
    assert handler.starts == 1
    assert handler.ends == 1
    assert handler.errors == 0


def test_run_with_callback_not_verbose() -> None:
    """Test run method works when callback manager is passed and not verbose."""
    import langchain

    langchain.verbose = False

    handler = FakeCallbackHandler()
    chain = FakeChain(callback_manager=CallbackManager(handlers=[handler]))
    output = chain.run("bar")
    assert output == "baz"
    assert handler.starts == 0
    assert handler.ends == 0
    assert handler.errors == 0
