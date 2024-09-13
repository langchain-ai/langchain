"""Test Cleanlab API wrapper.

In order to run this test, you need to have a Cleanlab Studio API key.
You can get it by registering for free at https://app.cleanlab.ai
Set the ``CLEANLAB_API_KEY`` environment variable with the above API key."""

from langchain_community.llms.cleanlab import CleanlabTLM


def test_cleanlab_call() -> None:
    """Test valid call to Cleanlab."""
    llm = CleanlabTLM(quality_preset="best")  # type: ignore[call-arg]
    output = llm.invoke("Say foo:")
    assert llm._llm_type == "cleanlab"
    assert isinstance(output, str)


async def test_cleanlab_acall() -> None:
    """Test aysync call to Cleanlab."""
    llm = CleanlabTLM()
    output = await llm.ainvoke("Say foo:")

    assert llm._llm_type == "cleanlab"
    assert isinstance(output, str)


def test_cleanlab_generate() -> None:
    """Test valid generation call to Cleanlab."""
    llm = CleanlabTLM()
    output = llm.generate(["Say foo:"])

    assert llm._llm_type == "cleanlab"
    assert isinstance(output, LLMResult)
    assert isinstance(output.generations[0][0].text, str)
    assert isinstance(
        output.generations[0][0].generation_info["trustworthiness_score"], float
    )
