"""Combined prompt injection tests - shows baseline vs protected for each model.

Each test runs both:
1. Baseline (no protection) - did the model trigger malicious tools?
2. Protected (with middleware) - was the injection blocked?

This gives a clear before/after picture of the middleware's effectiveness.
"""

import pytest

from .conftest import (
    INJECTION_TEST_CASES,
    check_vulnerability,
    create_combined_strategy,
    create_tool_message,
    create_tool_request,
)


def run_injection_test(model, payload, tools, tool_name, assertion, target_tools, model_name):
    """Run both baseline and protected tests, return results."""
    # Baseline: check if model is vulnerable without protection
    vulnerable, triggered = check_vulnerability(model, tools, payload, target_tools)
    
    # Protected: run with middleware
    strategy = create_combined_strategy(model, tools)
    req = create_tool_request(tools, tool_name)
    result = strategy.process(req, create_tool_message(payload, tool_name))
    protected_output = str(result.content)
    
    # Check if protected output passes assertion
    protected_safe = True
    try:
        assertion(protected_output)
    except AssertionError:
        protected_safe = False
    
    return {
        "model": model_name,
        "baseline_vulnerable": vulnerable,
        "baseline_triggered": triggered,
        "protected_safe": protected_safe,
    }


class TestOpenAI:
    """Prompt injection tests for OpenAI models."""

    @pytest.mark.requires("langchain_openai")
    @pytest.mark.parametrize("payload,tools,tool_name,assertion,target_tools", INJECTION_TEST_CASES)
    def test_injection(self, openai_model, payload, tools, tool_name, assertion, target_tools):
        r = run_injection_test(
            openai_model, payload, tools, tool_name, assertion, target_tools,
            model_name=openai_model.model_name
        )
        baseline = "VULN" if r["baseline_vulnerable"] else "safe"
        protected = "SAFE" if r["protected_safe"] else "FAIL"
        print(f"\n{r['model']}: baseline={baseline} triggered={r['baseline_triggered']} -> protected={protected}")
        assert r["protected_safe"], f"Protection failed for {r['model']}"


class TestAnthropic:
    """Prompt injection tests for Anthropic models."""

    @pytest.mark.requires("langchain_anthropic")
    @pytest.mark.parametrize("payload,tools,tool_name,assertion,target_tools", INJECTION_TEST_CASES)
    def test_injection(self, anthropic_model, payload, tools, tool_name, assertion, target_tools):
        r = run_injection_test(
            anthropic_model, payload, tools, tool_name, assertion, target_tools,
            model_name=anthropic_model.model
        )
        baseline = "VULN" if r["baseline_vulnerable"] else "safe"
        protected = "SAFE" if r["protected_safe"] else "FAIL"
        print(f"\n{r['model']}: baseline={baseline} triggered={r['baseline_triggered']} -> protected={protected}")
        assert r["protected_safe"], f"Protection failed for {r['model']}"


class TestOllama:
    """Prompt injection tests for Ollama models."""

    @pytest.mark.requires("langchain_ollama")
    @pytest.mark.parametrize("payload,tools,tool_name,assertion,target_tools", INJECTION_TEST_CASES)
    def test_injection(self, ollama_model, payload, tools, tool_name, assertion, target_tools):
        r = run_injection_test(
            ollama_model, payload, tools, tool_name, assertion, target_tools,
            model_name=ollama_model.model
        )
        baseline = "VULN" if r["baseline_vulnerable"] else "safe"
        protected = "SAFE" if r["protected_safe"] else "FAIL"
        print(f"\n{r['model']}: baseline={baseline} triggered={r['baseline_triggered']} -> protected={protected}")
        assert r["protected_safe"], f"Protection failed for {r['model']}"
