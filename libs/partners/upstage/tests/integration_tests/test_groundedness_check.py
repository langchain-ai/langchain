import os

import openai
import pytest

from langchain_upstage import GroundednessCheck


def test_langchain_upstage_groundedness_check() -> None:
    """Test Upstage Groundedness Check."""
    tool = GroundednessCheck()
    output = tool.run({"context": "foo bar", "query": "bar foo"})

    assert output in ["grounded", "notGrounded", "notSure"]

    api_key = os.environ.get("UPSTAGE_API_KEY", None)

    tool = GroundednessCheck(upstage_api_key=api_key)
    output = tool.run({"context": "foo bar", "query": "bar foo"})

    assert output in ["grounded", "notGrounded", "notSure"]


def test_langchain_upstage_groundedness_check_fail_with_wrong_api_key() -> None:
    tool = GroundednessCheck(api_key="wrong-key")
    with pytest.raises(openai.AuthenticationError):
        tool.run({"context": "foo bar", "query": "bar foo"})


async def test_langchain_upstage_groundedness_check_async() -> None:
    """Test Upstage Groundedness Check asynchronous."""
    tool = GroundednessCheck()
    output = await tool.arun({"context": "foo bar", "query": "bar foo"})

    assert output in ["grounded", "notGrounded", "notSure"]
