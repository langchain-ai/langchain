import os

import openai
import pytest
from langchain_core.documents import Document

from langchain_upstage import GroundednessCheck, UpstageGroundednessCheck


def test_langchain_upstage_groundedness_check_deprecated() -> None:
    """Test Upstage Groundedness Check."""
    tool = GroundednessCheck()
    output = tool.invoke({"context": "foo bar", "answer": "bar foo"})

    assert output in ["grounded", "notGrounded", "notSure"]

    api_key = os.environ.get("UPSTAGE_API_KEY", None)

    tool = GroundednessCheck(upstage_api_key=api_key)
    output = tool.invoke({"context": "foo bar", "answer": "bar foo"})

    assert output in ["grounded", "notGrounded", "notSure"]


def test_langchain_upstage_groundedness_check() -> None:
    """Test Upstage Groundedness Check."""
    tool = UpstageGroundednessCheck()
    output = tool.invoke({"context": "foo bar", "answer": "bar foo"})

    assert output in ["grounded", "notGrounded", "notSure"]

    api_key = os.environ.get("UPSTAGE_API_KEY", None)

    tool = UpstageGroundednessCheck(upstage_api_key=api_key)
    output = tool.invoke({"context": "foo bar", "answer": "bar foo"})

    assert output in ["grounded", "notGrounded", "notSure"]


def test_langchain_upstage_groundedness_check_with_documents_input() -> None:
    """Test Upstage Groundedness Check."""
    tool = UpstageGroundednessCheck()
    docs = [
        Document(page_content="foo bar"),
        Document(page_content="bar foo"),
    ]
    output = tool.invoke({"context": docs, "answer": "bar foo"})

    assert output in ["grounded", "notGrounded", "notSure"]


def test_langchain_upstage_groundedness_check_fail_with_wrong_api_key() -> None:
    tool = UpstageGroundednessCheck(api_key="wrong-key")
    with pytest.raises(openai.AuthenticationError):
        tool.invoke({"context": "foo bar", "answer": "bar foo"})


async def test_langchain_upstage_groundedness_check_async() -> None:
    """Test Upstage Groundedness Check asynchronous."""
    tool = UpstageGroundednessCheck()
    output = await tool.ainvoke({"context": "foo bar", "answer": "bar foo"})

    assert output in ["grounded", "notGrounded", "notSure"]
