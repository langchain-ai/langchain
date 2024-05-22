"""Test GooseAI API wrapper."""

from langchain_community.llms.gooseai import GooseAI


def test_gooseai_call() -> None:
    """Test valid call to gooseai."""
    llm = GooseAI(max_tokens=10)  # type: ignore[call-arg]
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)


def test_gooseai_call_fairseq() -> None:
    """Test valid call to gooseai with fairseq model."""
    llm = GooseAI(model_name="fairseq-1-3b", max_tokens=10)  # type: ignore[call-arg]
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)


def test_gooseai_stop_valid() -> None:
    """Test gooseai stop logic on valid configuration."""
    query = "write an ordered list of five items"
    first_llm = GooseAI(stop="3", temperature=0)  # type: ignore[call-arg]
    first_output = first_llm.invoke(query)
    second_llm = GooseAI(temperature=0)  # type: ignore[call-arg]
    second_output = second_llm.invoke(query, stop=["3"])
    # Because it stops on new lines, shouldn't return anything
    assert first_output == second_output
