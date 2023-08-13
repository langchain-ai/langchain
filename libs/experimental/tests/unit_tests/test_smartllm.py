"""Test SmartLLM."""
from langchain.chat_models import FakeListChatModel
from langchain.llms import FakeListLLM
from langchain.prompts.prompt import PromptTemplate

from langchain_experimental.smart_llm import SmartLLMChain


def test_ideation() -> None:
    # test that correct responses are returned
    responses = ["Idea 1", "Idea 2", "Idea 3"]
    llm = FakeListLLM(responses=responses)
    prompt = PromptTemplate(
        input_variables=["product"],
        template="What is a good name for a company that makes {product}?",
    )
    chain = SmartLLMChain(llm=llm, prompt=prompt)
    prompt_value, _ = chain.prep_prompts({"product": "socks"})
    chain.history.question = prompt_value.to_string()
    results = chain._ideate()
    assert results == responses

    # test that correct number of responses are returned
    for i in range(1, 5):
        responses = [f"Idea {j+1}" for j in range(i)]
        llm = FakeListLLM(responses=responses)
        chain = SmartLLMChain(llm=llm, prompt=prompt, n_ideas=i)
        prompt_value, _ = chain.prep_prompts({"product": "socks"})
        chain.history.question = prompt_value.to_string()
        results = chain._ideate()
        assert len(results) == i


def test_critique() -> None:
    response = "Test Critique"
    llm = FakeListLLM(responses=[response])
    prompt = PromptTemplate(
        input_variables=["product"],
        template="What is a good name for a company that makes {product}?",
    )
    chain = SmartLLMChain(llm=llm, prompt=prompt, n_ideas=2)
    prompt_value, _ = chain.prep_prompts({"product": "socks"})
    chain.history.question = prompt_value.to_string()
    chain.history.ideas = ["Test Idea 1", "Test Idea 2"]
    result = chain._critique()
    assert result == response


def test_resolver() -> None:
    response = "Test resolution"
    llm = FakeListLLM(responses=[response])
    prompt = PromptTemplate(
        input_variables=["product"],
        template="What is a good name for a company that makes {product}?",
    )
    chain = SmartLLMChain(llm=llm, prompt=prompt, n_ideas=2)
    prompt_value, _ = chain.prep_prompts({"product": "socks"})
    chain.history.question = prompt_value.to_string()
    chain.history.ideas = ["Test Idea 1", "Test Idea 2"]
    chain.history.critique = "Test Critique"
    result = chain._resolve()
    assert result == response


def test_all_steps() -> None:
    joke = "Why did the chicken cross the Mobius strip?"
    response = "Resolution response"
    ideation_llm = FakeListLLM(responses=["Ideation response" for _ in range(20)])
    critique_llm = FakeListLLM(responses=["Critique response" for _ in range(20)])
    resolver_llm = FakeListLLM(responses=[response for _ in range(20)])
    prompt = PromptTemplate(
        input_variables=["joke"],
        template="Explain this joke to me: {joke}?",
    )
    chain = SmartLLMChain(
        ideation_llm=ideation_llm,
        critique_llm=critique_llm,
        resolver_llm=resolver_llm,
        prompt=prompt,
    )
    result = chain(joke)
    assert result["joke"] == joke
    assert result["resolution"] == response


def test_intermediate_output() -> None:
    joke = "Why did the chicken cross the Mobius strip?"
    llm = FakeListLLM(responses=[f"Response {i+1}" for i in range(5)])
    prompt = PromptTemplate(
        input_variables=["joke"],
        template="Explain this joke to me: {joke}?",
    )
    chain = SmartLLMChain(llm=llm, prompt=prompt, return_intermediate_steps=True)
    result = chain(joke)
    assert result["joke"] == joke
    assert result["ideas"] == [f"Response {i+1}" for i in range(3)]
    assert result["critique"] == "Response 4"
    assert result["resolution"] == "Response 5"


def test_all_steps_with_chat_model() -> None:
    joke = "Why did the chicken cross the Mobius strip?"
    response = "Resolution response"

    ideation_llm = FakeListChatModel(responses=["Ideation response" for _ in range(20)])
    critique_llm = FakeListChatModel(responses=["Critique response" for _ in range(20)])
    resolver_llm = FakeListChatModel(responses=[response for _ in range(20)])
    prompt = PromptTemplate(
        input_variables=["joke"],
        template="Explain this joke to me: {joke}?",
    )
    chain = SmartLLMChain(
        ideation_llm=ideation_llm,
        critique_llm=critique_llm,
        resolver_llm=resolver_llm,
        prompt=prompt,
    )
    result = chain(joke)
    assert result["joke"] == joke
    assert result["resolution"] == response
