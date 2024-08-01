"""Test sambanova API wrapper.

In order to run this test, you need to have an sambaverse api key,
and a sambaverse base url, project id, endpoint id, and api key.
You'll then need to set SAMBAVERSE_API_KEY, SAMBASTUDIO_BASE_URL,
SAMBASTUDIO_PROJECT_ID, SAMBASTUDIO_ENDPOINT_ID, and SAMBASTUDIO_API_KEY
environment variables.
"""

from langchain_community.llms.sambanova import SambaStudio, Sambaverse


def test_sambaverse_call() -> None:
    """Test simple non-streaming call to sambaverse."""
    llm = Sambaverse(
        sambaverse_model_name="Meta/llama-2-7b-chat-hf",
        model_kwargs={"select_expert": "llama-2-7b-chat-hf"},
    )
    output = llm.invoke("What is LangChain")
    assert output
    assert isinstance(output, str)


def test_sambastudio_call() -> None:
    """Test simple non-streaming call to sambaverse."""
    llm = SambaStudio()
    output = llm.invoke("What is LangChain")
    assert output
    assert isinstance(output, str)
