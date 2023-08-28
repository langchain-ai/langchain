"""Integration test for self ask with search."""
from langchain_xfyun.agents.self_ask_with_search.base import SelfAskWithSearchChain
from langchain_xfyun.llms.openai import OpenAI
from langchain_xfyun.utilities.google_serper import GoogleSerperAPIWrapper


def test_self_ask_with_search() -> None:
    """Test functionality on a prompt."""
    question = "What is the hometown of the reigning men's U.S. Open champion?"
    chain = SelfAskWithSearchChain(
        llm=OpenAI(temperature=0),
        search_chain=GoogleSerperAPIWrapper(),
        input_key="q",
        output_key="a",
    )
    answer = chain.run(question)
    final_answer = answer.split("\n")[-1]
    assert final_answer == "El Palmar, Spain"
