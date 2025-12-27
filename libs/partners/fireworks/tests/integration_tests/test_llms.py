"""Test Fireworks API wrapper.

In order to run this test, you need to have an Fireworks api key.

You can get it by registering for free at https://api.fireworks.ai/.

A test key can be found at https://api.fireworks.ai/settings/api-keys

You'll then need to set `FIREWORKS_API_KEY` environment variable to your api key.
"""

import pytest as pytest

from langchain_fireworks import Fireworks

_MODEL = "accounts/fireworks/models/deepseek-v3p1"


def test_fireworks_call() -> None:
    """Test simple call to fireworks."""
    llm = Fireworks(
        model=_MODEL,
        temperature=0.2,
        max_tokens=250,
    )
    output = llm.invoke("Say foo:")

    assert llm._llm_type == "fireworks"
    assert isinstance(output, str)
    assert len(output) > 0


async def test_fireworks_acall() -> None:
    """Test simple call to fireworks."""
    llm = Fireworks(
        model=_MODEL,
        temperature=0.2,
        max_tokens=250,
    )
    output = await llm.agenerate(["Say foo:"], stop=["bar"])

    assert llm._llm_type == "fireworks"
    output_text = output.generations[0][0].text
    assert isinstance(output_text, str)
    assert output_text.count("bar") <= 1


def test_stream() -> None:
    """Test streaming tokens from OpenAI."""
    llm = Fireworks(model=_MODEL)

    for token in llm.stream("I'm Pickle Rick"):
        assert isinstance(token, str)


async def test_astream() -> None:
    """Test streaming tokens from OpenAI."""
    llm = Fireworks(model=_MODEL)

    async for token in llm.astream("I'm Pickle Rick"):
        assert isinstance(token, str)


async def test_abatch() -> None:
    """Test streaming tokens from Fireworks."""
    llm = Fireworks(model=_MODEL)

    result = await llm.abatch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token, str)


async def test_abatch_tags() -> None:
    """Test batch tokens from Fireworks."""
    llm = Fireworks(model=_MODEL)

    result = await llm.abatch(
        ["I'm Pickle Rick", "I'm not Pickle Rick"], config={"tags": ["foo"]}
    )
    for token in result:
        assert isinstance(token, str)


def test_batch() -> None:
    """Test batch tokens from Fireworks."""
    llm = Fireworks(model=_MODEL)

    result = llm.batch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token, str)


async def test_ainvoke() -> None:
    """Test invoke tokens from Fireworks."""
    llm = Fireworks(model=_MODEL)

    result = await llm.ainvoke("I'm Pickle Rick", config={"tags": ["foo"]})
    assert isinstance(result, str)


def test_invoke() -> None:
    """Test invoke tokens from Fireworks."""
    llm = Fireworks(model=_MODEL)

    result = llm.invoke("I'm Pickle Rick", config={"tags": ["foo"]})
    assert isinstance(result, str)
