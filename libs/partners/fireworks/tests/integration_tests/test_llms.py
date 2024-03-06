"""Test Fireworks API wrapper.

In order to run this test, you need to have an Fireworks api key.
You can get it by registering for free at https://api.fireworks.ai/.
A test key can be found at https://api.fireworks.ai/settings/api-keys

You'll then need to set FIREWORKS_API_KEY environment variable to your api key.
"""

import pytest as pytest

from langchain_fireworks import Fireworks


def test_fireworks_call() -> None:
    """Test simple call to fireworks."""
    llm = Fireworks(
        model="accounts/fireworks/models/mixtral-8x7b-instruct",
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
        model="accounts/fireworks/models/mixtral-8x7b-instruct",
        temperature=0.2,
        max_tokens=250,
    )
    output = await llm.agenerate(["Say foo:"], stop=["bar"])

    assert llm._llm_type == "fireworks"
    output_text = output.generations[0][0].text
    assert isinstance(output_text, str)
    assert output_text.count("bar") <= 1
