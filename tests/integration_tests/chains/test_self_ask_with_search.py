"""Integration test for self ask with search."""
from langchain.chains.self_ask_with_search.base import SelfAskWithSearchChain
from langchain.chains.serpapi import SerpAPIChain
from langchain.llms.openai import OpenAI


def test_self_ask_with_search() -> None:
    """Test functionality on a prompt."""
    question = "What is the hometown of the reigning men's U.S. Open champion?"
    chain = SelfAskWithSearchChain(
        llm=OpenAI(temperature=0),
        search_chain=SerpAPIChain(),
        input_key="q",
        output_key="a",
    )
    answer = chain.run(question)
    final_answer = answer.split("\n")[-1]
    assert final_answer == "So the final answer is: El Palmar, Murcia, Spain"
