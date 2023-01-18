"""Test Anthropic API wrapper."""

from pathlib import Path
from typing import Generator

import pytest

from langchain.llms.loading import load_llm
from langchain.llms.anthropic import Anthropic


def test_anthropic_call() -> None:
    """Test valid call to anthropic."""
    llm = Anthropic(model="bare-nano-0")
    output = llm("Say foo:")
    print(output)
    assert isinstance(output, str)


def test_anthropic_streaming() -> None:
    """Test streaming tokens from anthropic."""
    llm = Anthropic(model="bare-nano-0")
    generator = llm.stream("I'm Pickle Rick")

    assert isinstance(generator, Generator)

    for token in generator:
        print(token)
        assert isinstance(token["completion"], str)
