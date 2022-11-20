"""Test logic on base chain class."""
from typing import Dict, List

import pytest
from pydantic import BaseModel

from langchain.chains.base import Chain


class FakeChain(Chain, BaseModel):
    """Fake chain class for testing purposes."""

    be_correct: bool = True

    @property
    def input_keys(self) -> List[str]:
        """Input key of foo."""
        return ["foo"]

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
