"""Test sambanova API wrapper.

In order to run this test, you need to have a sambastudio base url,
project id, endpoint id, and api key.
You'll then need to set SAMBASTUDIO_BASE_URL, SAMBASTUDIO_BASE_URI
SAMBASTUDIO_PROJECT_ID, SAMBASTUDIO_ENDPOINT_ID, and SAMBASTUDIO_API_KEY
environment variables.
"""

from langchain_community.llms.sambanova import SambaStudio


def test_sambastudio_call() -> None:
    """Test simple non-streaming call to sambastudio."""
    llm = SambaStudio()
    output = llm.invoke("What is LangChain")
    assert output
    assert isinstance(output, str)
