"""Test sambanova API llm wrappers.

In order to run this test, you need to have a sambastudio url, and api key
and a sambanova cloud api key.
You'll then need to set SAMBASTUDIO_URL, and SAMBASTUDIO_API_KEY,
and SAMBANOVA_API_KEY environment variables.
"""

from langchain_community.llms.sambanova import SambaNovaCloud, SambaStudio


def test_sambanova_cloud_call() -> None:
    """Test simple non-streaming call to sambastudio."""
    llm = SambaNovaCloud()
    output = llm.invoke("What is LangChain")
    assert output
    assert isinstance(output, str)


def test_sambastudio_call() -> None:
    """Test simple non-streaming call to sambastudio."""
    llm = SambaStudio()
    output = llm.invoke("What is LangChain")
    assert output
    assert isinstance(output, str)
