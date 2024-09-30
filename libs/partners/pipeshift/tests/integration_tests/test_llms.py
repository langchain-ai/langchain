"""Testing Pipeshift API wrapper.

pre-requisites: Pipeshift API key
- to get one, go ahead and create an account at https://pipeshift.com/
- then add your api key to environment variables,
- PIPESHIFT_API_KEY=<your_pipeshift_api_key>
"""

import pytest as pytest  # type: ignore[import-not-found]

from langchain_pipeshift import Pipeshift


def test_pipeshift_call() -> None:
    """Test simple call to pipeshift."""
    llm = Pipeshift(  # type: ignore[call-arg]
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        temperature=0.2,
        max_tokens=250,
    )
    output = llm.invoke("Say foo:")

    assert llm._llm_type == "pipeshift"
    assert isinstance(output, str)
    assert len(output) > 0


async def test_pipeshift_acall() -> None:
    """Test simple call to pipeshift."""
    llm = Pipeshift(  # type: ignore[call-arg]
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        temperature=0.2,
        max_tokens=250,
    )
    output = await llm.agenerate(["Say foo:"], stop=["bar"])

    assert llm._llm_type == "pipeshift"
    output_text = output.generations[0][0].text
    assert isinstance(output_text, str)
    assert output_text.count("bar") <= 1
