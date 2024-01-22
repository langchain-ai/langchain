"""Test AI21LLM llm."""
import pytest
from ai21.models import Penalty

from langchain_ai21.llms import AI21


def test_stream() -> None:
    """Test streaming tokens from OpenAI."""
    llm = AI21()

    with pytest.raises(NotImplementedError):
        for token in llm.stream("I'm Pickle Rick"):
            assert isinstance(token, str)


async def test_astream() -> None:
    """Test streaming tokens from OpenAI."""
    llm = AI21()

    async for token in llm.astream("I'm Pickle Rick"):
        assert isinstance(token, str)


async def test_abatch() -> None:
    """Test streaming tokens from AI21LLM."""
    llm = AI21()

    result = await llm.abatch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token, str)


async def test_abatch_tags() -> None:
    """Test batch tokens from AI21LLM."""
    llm = AI21()

    result = await llm.abatch(
        ["I'm Pickle Rick", "I'm not Pickle Rick"], config={"tags": ["foo"]}
    )
    for token in result:
        assert isinstance(token, str)


def test_batch() -> None:
    """Test batch tokens from AI21LLM."""
    llm = AI21()

    result = llm.batch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token, str)


async def test_ainvoke() -> None:
    """Test invoke tokens from AI21LLM."""
    llm = AI21()

    result = await llm.ainvoke("I'm Pickle Rick", config={"tags": ["foo"]})
    assert isinstance(result, str)


def test_invoke() -> None:
    """Test invoke tokens from AI21LLM."""
    llm = AI21()

    result = llm.invoke("I'm Pickle Rick", config=dict(tags=["foo"]))
    assert isinstance(result, str)


def test__generate() -> None:
    llm = AI21(
        max_tokens=2,
        temperature=0,
        top_p=1,
        top_k_return=0,
        num_results=1,
        epoch=1,
        count_penalty=Penalty(
            scale=0,
            apply_to_emojis=False,
            apply_to_numbers=False,
            apply_to_stopwords=False,
            apply_to_punctuation=False,
            apply_to_whitespaces=False,
        ),
        frequency_penalty=Penalty(
            scale=0,
            apply_to_emojis=False,
            apply_to_numbers=False,
            apply_to_stopwords=False,
            apply_to_punctuation=False,
            apply_to_whitespaces=False,
        ),
        presence_penalty=Penalty(
            scale=0,
            apply_to_emojis=False,
            apply_to_numbers=False,
            apply_to_stopwords=False,
            apply_to_punctuation=False,
            apply_to_whitespaces=False,
        ),
    )
    llm_result = llm.generate(
        prompts=["Hey there, my name is Pickle Rick. What is your name?"],
        stop=["##"],
    )

    assert len(llm_result.generations) > 0
    assert llm_result.llm_output["token_count"] != 0
