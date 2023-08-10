"""Test EdenAi API wrapper.

In order to run this test, you need to have an EdenAI api key.
You can get it by registering for free at https://app.edenai.run/user/register.
A test key can be found at https://app.edenai.run/admin/account/settings by
clicking on the 'sandbox' toggle.
(calls will be free, and will return dummy results)

You'll then need to set EDENAI_API_KEY environment variable to your api key.
"""
from langchain.llms import EdenAI


def test_edenai_call() -> None:
    """Test simple call to edenai."""
    llm = EdenAI(provider="openai", params={"temperature": 0.2, "max_tokens": 250})
    output = llm("Say foo:")

    assert llm._llm_type == "edenai"
    assert llm.feature == "text"
    assert llm.subfeature == "generation"
    assert isinstance(output, str)


async def test_edenai_acall() -> None:
    """Test simple call to edenai."""
    llm = EdenAI(provider="openai", params={"temperature": 0.2, "max_tokens": 250})
    output = await llm.agenerate(["Say foo:"])
    assert llm._llm_type == "edenai"
    assert llm.feature == "text"
    assert llm.subfeature == "generation"
    assert isinstance(output, str)
