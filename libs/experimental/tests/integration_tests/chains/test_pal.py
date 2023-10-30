"""Test PAL chain."""

from langchain.llms import OpenAI

from langchain_experimental.pal_chain.base import PALChain


def test_math_prompt() -> None:
    """Test math prompt."""
    llm = OpenAI(temperature=0, max_tokens=512)
    pal_chain = PALChain.from_math_prompt(llm, timeout=None)
    question = (
        "Jan has three times the number of pets as Marcia. "
        "Marcia has two more pets than Cindy. "
        "If Cindy has four pets, how many total pets do the three have?"
    )
    output = pal_chain.run(question)
    assert output == "28"


def test_colored_object_prompt() -> None:
    """Test colored object prompt."""
    llm = OpenAI(temperature=0, max_tokens=512)
    pal_chain = PALChain.from_colored_object_prompt(llm, timeout=None)
    question = (
        "On the desk, you see two blue booklets, "
        "two purple booklets, and two yellow pairs of sunglasses. "
        "If I remove all the pairs of sunglasses from the desk, "
        "how many purple items remain on it?"
    )
    output = pal_chain.run(question)
    assert output == "2"
