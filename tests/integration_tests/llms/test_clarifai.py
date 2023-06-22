"""Test Clarifai API wrapper.
In order to run this test, you need to have an account on Clarifai.
You can sign up for free at https://clarifai.com/signup.
pip install clarifai

You'll need to set env variable CLARIFAI_PAT_KEY to your personal access token key.
"""

from langchain.llms.clarifai import Clarifai


def test_clarifai_call() -> None:
    """Test valid call to clarifai."""
    llm = Clarifai(
        user_id="google-research",
        app_id="summarization",
        model_id="text-summarization-english-pegasus",
    )
    output = llm(
        "A chain is a serial assembly of connected pieces, called links, \
        typically made of metal, with an overall character similar to that\
        of a rope in that it is flexible and curved in compression but \
        linear, rigid, and load-bearing in tension. A chain may consist\
        of two or more links."
    )

    assert isinstance(output, str)
    assert llm._llm_type == "clarifai"
    assert llm.model_id == "text-summarization-english-pegasus"
