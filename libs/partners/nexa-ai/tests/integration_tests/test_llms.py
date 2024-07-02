"""Test NexaAILLM llm."""
from langchain_core.outputs import Generation, LLMResult

from langchain_nexa_ai import NexaAILLM
from tests import temporary_api_key


def test_invoke() -> None:
    """Test invoke tokens from NexaAILLM."""
    with temporary_api_key():
        llm = NexaAILLM()

    result = llm.invoke(
        "Show recommended products for electronics.", category="shopping"
    )

    assert "result" in result
    assert "latency" in result
    assert "function_name" in result
    assert "function_arguments" in result
    assert isinstance(result, str)


def test_generate() -> None:
    """Test sync generate."""
    with temporary_api_key():
        llm = NexaAILLM()
    response = llm.generate(
        prompts=["Show recommended products for electronics.", "Hotels near me."],
        categories=["shopping", "travel"],
    )
    assert isinstance(response, LLMResult)
    assert len(response.generations) == 2
    assert response.llm_output
    for generations in response.generations:
        assert len(generations) == 1
        for generation in generations:
            assert isinstance(generation, Generation)
            assert isinstance(generation.text, str)
            assert isinstance(generation.generation_info, dict)
            assert "result" in generation.generation_info
            assert "latency" in generation.generation_info
            assert "function_name" in generation.generation_info
            assert "function_arguments" in generation.generation_info
