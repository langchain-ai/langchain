"""Testing Pipeshift API wrapper.

pre-requisites: Pipeshift API key
- to get one, go ahead and create an account at https://pipeshift.com/
- then add your api key to environment variables,
- PIPESHIFT_API_KEY=<your_pipeshift_api_key>
"""

import pytest as pytest  # type: ignore[import-not-found]

from langchain_community.llms.pipeshift import Pipeshift

DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"


@pytest.mark.enable_socket
def test_pipeshift_call() -> None:
    """Test simple call to pipeshift."""
    llm = Pipeshift(  # type: ignore[call-arg]
        model="mistralai/Mistral-7B-Instruct-v0.3",
        temperature=0.2,
        max_tokens=250,
    )
    output = llm.invoke("Say foo:")

    assert llm._llm_type == "pipeshift"
    assert isinstance(output, str)
    assert len(output) > 0
