"""Test SmartLLM."""
import asyncio

from langchain.llms import FakeListLLM
from langchain.llms.base import SmartLLM

# Test:
# - critique: returns str
# - resolver: returns str
# - add history 1
# - add history 2
# - async


def test_ideation() -> None:
    # test that correct reponses are returned
    responses = ["Idea 1", "Idea 2", "Idea 3"]
    llm = FakeListLLM(responses=responses)
    smart_llm = SmartLLM(llm=llm)
    results = smart_llm._ideate("example prompt")
    assert results == responses

    # test that correct number of responses are returned
    for i in range(1, 5):
        responses = [f"Idea {j+1}" for j in range(i)]
        llm = FakeListLLM(responses=responses)
        smart_llm = SmartLLM(llm=llm, n_ideas=i)
        results = smart_llm._ideate("example prompt")
        assert len(results) == i


def test_critique() -> None:
    responses = ["Idea 1"]
    llm = FakeListLLM(responses=responses)
    smart_llm = SmartLLM(llm=llm)
    result = smart_llm._critique()
    assert result == responses[0]


def test_resolver() -> None:
    responses = ["Idea 1"]
    llm = FakeListLLM(responses=responses)
    smart_llm = SmartLLM(llm=llm)
    result = smart_llm._resolve()
    assert result == responses[0]


def test_add_history() -> None:
    responses = ["Idea 1", "Idea 2", "Idea 3"]
    llm = FakeListLLM(responses=responses)
    smart_llm = SmartLLM(llm=llm)

    question = "What is LangChain?"
    ideas = ["A luxury brand", "An open-source LMM library", "A dog"]
    critique = "Why would a dog be called LangChain?"

    # test update_history_after_ideation
    smart_llm._update_history_after_ideation(question, ideas)
    assert question in smart_llm.history
    for idea in ideas:
        assert idea in smart_llm.history

    # test update_history_after_critique
    smart_llm._update_history_after_critique(critique)
    assert question in smart_llm.history
    for idea in ideas:
        assert idea in smart_llm.history
    assert critique in smart_llm.history


def test_all_steps() -> None:
    ideation_llm = FakeListLLM(responses=[f"Fake ideation response" for _ in range(20)])
    critique_llm = FakeListLLM(responses=[f"Fake critique response" for _ in range(20)])
    resolver_llm = FakeListLLM(
        responses=[f"Fake resolution response" for _ in range(20)]
    )
    smart_llm = SmartLLM(
        ideation_llm=ideation_llm, critique_llm=critique_llm, resolver_llm=resolver_llm
    )
    result = smart_llm("Why did the chicken cross the Mobius strip?")
    assert result == "Fake resolution response"
