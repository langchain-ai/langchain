"""Extended tests for prompt injection defense using real LLM models."""

import pytest

from .conftest import (
    INJECTION_TEST_CASES,
    create_combined_strategy,
    create_tool_message,
    create_tool_request,
)


class TestOpenAI:
    """Prompt injection defense tests for OpenAI models."""

    @pytest.mark.requires("langchain_openai")
    @pytest.mark.parametrize("payload,tools,tool_name,assertion,_target", INJECTION_TEST_CASES)
    def test_injection_blocked(self, openai_model, payload, tools, tool_name, assertion, _target):
        strategy = create_combined_strategy(openai_model, tools)
        req = create_tool_request(tools, tool_name)
        result = strategy.process(req, create_tool_message(payload, tool_name))
        assertion(str(result.content))


class TestAnthropic:
    """Prompt injection defense tests for Anthropic models."""

    @pytest.mark.requires("langchain_anthropic")
    @pytest.mark.parametrize("payload,tools,tool_name,assertion,_target", INJECTION_TEST_CASES)
    def test_injection_blocked(self, anthropic_model, payload, tools, tool_name, assertion, _target):
        strategy = create_combined_strategy(anthropic_model, tools)
        req = create_tool_request(tools, tool_name)
        result = strategy.process(req, create_tool_message(payload, tool_name))
        assertion(str(result.content))


class TestOllama:
    """Prompt injection defense tests for Ollama models."""

    @pytest.mark.requires("langchain_ollama")
    @pytest.mark.parametrize("payload,tools,tool_name,assertion,_target", INJECTION_TEST_CASES)
    def test_injection_blocked(self, ollama_model, payload, tools, tool_name, assertion, _target):
        strategy = create_combined_strategy(ollama_model, tools)
        req = create_tool_request(tools, tool_name)
        result = strategy.process(req, create_tool_message(payload, tool_name))
        assertion(str(result.content))
