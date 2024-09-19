"""Test Konko API wrapper.

In order to run this test, you need to have an Konko api key.
You'll then need to set KONKO_API_KEY environment variable to your api key.
"""

import pytest as pytest

from langchain_community.llms import Konko


def test_konko_call() -> None:
    """Test simple call to konko."""
    llm = Konko(
        model="mistralai/mistral-7b-v0.1",
        temperature=0.2,
        max_tokens=250,
    )
    output = llm.invoke("Say foo:")

    assert llm._llm_type == "konko"
    assert isinstance(output, str)


async def test_konko_acall() -> None:
    """Test simple call to konko."""
    llm = Konko(
        model="mistralai/mistral-7b-v0.1",
        temperature=0.2,
        max_tokens=250,
    )
    output = await llm.agenerate(["Say foo:"], stop=["bar"])

    assert llm._llm_type == "konko"
    output_text = output.generations[0][0].text
    assert isinstance(output_text, str)
    assert output_text.count("bar") <= 1
