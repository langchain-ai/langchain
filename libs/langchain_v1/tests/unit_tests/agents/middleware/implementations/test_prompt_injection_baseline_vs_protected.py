"""Baseline vs Protected prompt injection tests.

Compares model behavior with and without defense middleware:
1. Baseline (no protection) - does the model trigger malicious tools?
2. Protected (with middleware) - is the injection blocked?

This demonstrates the middleware's effectiveness at preventing prompt injection.

NOTE: These tests are skipped by default in CI because they:
1. Make real API calls to LLM providers (costs money)
2. Are slow (multiple LLM roundtrips per test)
3. Are for manual validation, not regression testing

To run manually:
    RUN_BENCHMARK_TESTS=1 pytest test_prompt_injection_baseline_vs_protected.py -v -s
"""

import os

import pytest

# Skip all tests in this module by default unless explicitly enabled
pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_BENCHMARK_TESTS") != "1",
    reason="E2E tests are skipped by default. Set RUN_BENCHMARK_TESTS=1 to run.",
)

from .conftest import (
    INJECTION_TEST_CASES,
    check_vulnerability,
    create_combined_strategy,
    create_tool_message,
    create_tool_request,
)


def _get_model_name(model):
    """Extract model name from different model types."""
    return getattr(model, "model_name", None) or getattr(model, "model", "unknown")


def _run_baseline_vs_protected(model, payload, tools, tool_name, assertion, target_tools):
    """Run both baseline and protected tests, return results."""
    model_name = _get_model_name(model)

    # Baseline: check if model is vulnerable without protection
    vulnerable, triggered = check_vulnerability(model, tools, payload, target_tools)

    # Protected: run with middleware
    strategy = create_combined_strategy(model, tools)
    req = create_tool_request(tools, tool_name)
    result = strategy.process(req, create_tool_message(payload, tool_name))

    # Check if protected output passes assertion
    protected_safe = True
    try:
        assertion(str(result.content))
    except AssertionError:
        protected_safe = False

    # Print comparison
    baseline = "VULN" if vulnerable else "safe"
    protected = "SAFE" if protected_safe else "FAIL"
    print(f"\n{model_name}: baseline={baseline} triggered={triggered} -> protected={protected}")

    return protected_safe, model_name


class TestOpenAI:
    """Baseline vs protected tests for OpenAI models (gpt-5.2)."""

    @pytest.mark.requires("langchain_openai")
    @pytest.mark.parametrize("payload,tools,tool_name,assertion,target_tools", INJECTION_TEST_CASES)
    def test_injection(self, openai_model, payload, tools, tool_name, assertion, target_tools):
        protected_safe, model_name = _run_baseline_vs_protected(
            openai_model, payload, tools, tool_name, assertion, target_tools
        )
        assert protected_safe, f"Protection failed for {model_name}"


class TestAnthropic:
    """Baseline vs protected tests for Anthropic models (claude-opus-4-5)."""

    @pytest.mark.requires("langchain_anthropic")
    @pytest.mark.parametrize("payload,tools,tool_name,assertion,target_tools", INJECTION_TEST_CASES)
    def test_injection(self, anthropic_model, payload, tools, tool_name, assertion, target_tools):
        protected_safe, model_name = _run_baseline_vs_protected(
            anthropic_model, payload, tools, tool_name, assertion, target_tools
        )
        assert protected_safe, f"Protection failed for {model_name}"


class TestOllama:
    """Baseline vs protected tests for Ollama models (granite4:tiny-h)."""

    @pytest.mark.requires("langchain_ollama")
    @pytest.mark.parametrize("payload,tools,tool_name,assertion,target_tools", INJECTION_TEST_CASES)
    def test_injection(self, ollama_model, payload, tools, tool_name, assertion, target_tools):
        protected_safe, model_name = _run_baseline_vs_protected(
            ollama_model, payload, tools, tool_name, assertion, target_tools
        )
        assert protected_safe, f"Protection failed for {model_name}"
