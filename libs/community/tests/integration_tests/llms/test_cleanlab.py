"""Test Cleanlab API for Trustworthy Language Model.

In order to run this test, you need to have a Cleanlab Studio API key.
You can get it by registering for free at https://app.cleanlab.ai
Set the ``CLEANLAB_API_KEY`` environment variable with the above API key."""

from langchain_community.llms.cleanlab import TrustworthyLanguageModel
from langchain_core.outputs import LLMResult


def test_cleanlab_call() -> None:
    """Test valid call to TLM."""
    tlm = TrustworthyLanguageModel(quality_preset="best")  # type: ignore[call-arg]
    output = tlm.invoke("Say foo:")
    assert tlm._llm_type == "trustworthy_language_model"
    assert isinstance(output, str)


async def test_cleanlab_acall() -> None:
    """Test aysync call to TLM."""
    tlm = TrustworthyLanguageModel()
    output = await tlm.ainvoke("Say foo:")

    assert tlm._llm_type == "trustworthy_language_model"
    assert isinstance(output, str)


def test_cleanlab_generate() -> None:
    """Test valid generation call to TLM."""
    tlm = TrustworthyLanguageModel()
    output = tlm.generate(["Say foo:"])

    assert tlm._llm_type == "trustworthy_language_model"
    assert isinstance(output, LLMResult)
    assert isinstance(output.generations[0][0].text, str)
    assert isinstance(
        output.generations[0][0].generation_info["trustworthiness_score"], float
    )