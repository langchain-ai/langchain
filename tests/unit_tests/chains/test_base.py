"""Test logic on base chain class."""
from typing import Dict, List

import pytest
from pydantic import BaseModel

from langchain.chains.base import Chain


class FakeChain(Chain, BaseModel):
    """Fake chain class for testing purposes."""

    be_correct: bool = True
    the_input_keys: List[str] = ["foo"]

    @property
    def input_keys(self) -> List[str]:
        """Input keys."""
        return self.the_input_keys

    @property
    def output_keys(self) -> List[str]:
        """Output key of bar."""
        return ["bar"]

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
