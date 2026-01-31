"""Argument hijacking prompt injection tests.

Tests a more subtle attack vector: the user legitimately wants to call a tool,
but the injection manipulates the arguments (e.g., changing email recipient,
appending malicious content to body, swapping URLs).

This is harder to detect than unauthorized tool triggering because:
1. The tool call itself is expected
2. Only the arguments are compromised
3. The attack blends with legitimate data

NOTE: These tests are skipped by default in CI because they:
1. Make real API calls to LLM providers (costs money)
2. Are slow (multiple LLM roundtrips per test)
3. Are for manual validation, not regression testing

To run manually:
    RUN_BENCHMARK_TESTS=1 pytest test_prompt_injection_arg_hijacking.py -v -s
"""

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_BENCHMARK_TESTS") != "1",
    reason="E2E tests are skipped by default. Set RUN_BENCHMARK_TESTS=1 to run.",
)

from .conftest import (
    ARG_HIJACK_TEST_CASES,
    check_argument_hijacking,
    create_combined_strategy,
    create_tool_message,
    create_tool_request,
)


def _get_model_name(model):
    """Extract model name from different model types."""
    return getattr(model, "model_name", None) or getattr(model, "model", "unknown")


def _run_arg_hijack_test(
    model, user_request, payload, tools, expected_tool, legitimate_args, malicious_patterns
):
    """Test argument hijacking with and without protection.

    Returns:
        (baseline_hijacked, protected_hijacked, model_name)
    """
    model_name = _get_model_name(model)

    # Baseline: check if model is vulnerable without protection
    baseline_hijacked, _, baseline_details = check_argument_hijacking(
        model, tools, user_request, payload, expected_tool, legitimate_args, malicious_patterns
    )

    # Protected: run payload through middleware first
    strategy = create_combined_strategy(model, tools)
    req = create_tool_request(tools, "get_user_data")
    sanitized = strategy.process(req, create_tool_message(payload, "get_user_data"))

    # Now check if sanitized payload still causes hijacking
    protected_hijacked, _, protected_details = check_argument_hijacking(
        model,
        tools,
        user_request,
        str(sanitized.content),
        expected_tool,
        legitimate_args,
        malicious_patterns,
    )

    # Print comparison
    baseline = "HIJACKED" if baseline_hijacked else "clean"
    protected = "SAFE" if not protected_hijacked else "HIJACKED"
    print(f"\n{model_name}: baseline={baseline} ({baseline_details})")
    print(f"  -> protected={protected} ({protected_details})")

    return baseline_hijacked, protected_hijacked, model_name


class TestOpenAI:
    """Argument hijacking tests for OpenAI models."""

    @pytest.mark.requires("langchain_openai")
    @pytest.mark.parametrize(
        (
            "user_request",
            "payload",
            "tools",
            "expected_tool",
            "legitimate_args",
            "malicious_patterns",
        ),
        ARG_HIJACK_TEST_CASES,
    )
    def test_arg_hijacking(
        self,
        openai_model,
        user_request,
        payload,
        tools,
        expected_tool,
        legitimate_args,
        malicious_patterns,
    ):
        _, protected_hijacked, model_name = _run_arg_hijack_test(
            openai_model,
            user_request,
            payload,
            tools,
            expected_tool,
            legitimate_args,
            malicious_patterns,
        )
        assert not protected_hijacked, f"Argument hijacking not prevented for {model_name}"


class TestAnthropic:
    """Argument hijacking tests for Anthropic models."""

    @pytest.mark.requires("langchain_anthropic")
    @pytest.mark.parametrize(
        (
            "user_request",
            "payload",
            "tools",
            "expected_tool",
            "legitimate_args",
            "malicious_patterns",
        ),
        ARG_HIJACK_TEST_CASES,
    )
    def test_arg_hijacking(
        self,
        anthropic_model,
        user_request,
        payload,
        tools,
        expected_tool,
        legitimate_args,
        malicious_patterns,
    ):
        _, protected_hijacked, model_name = _run_arg_hijack_test(
            anthropic_model,
            user_request,
            payload,
            tools,
            expected_tool,
            legitimate_args,
            malicious_patterns,
        )
        assert not protected_hijacked, f"Argument hijacking not prevented for {model_name}"


class TestGoogle:
    """Argument hijacking tests for Google models."""

    @pytest.mark.requires("langchain_google_genai")
    @pytest.mark.parametrize(
        (
            "user_request",
            "payload",
            "tools",
            "expected_tool",
            "legitimate_args",
            "malicious_patterns",
        ),
        ARG_HIJACK_TEST_CASES,
    )
    def test_arg_hijacking(
        self,
        google_model,
        user_request,
        payload,
        tools,
        expected_tool,
        legitimate_args,
        malicious_patterns,
    ):
        _, protected_hijacked, model_name = _run_arg_hijack_test(
            google_model,
            user_request,
            payload,
            tools,
            expected_tool,
            legitimate_args,
            malicious_patterns,
        )
        assert not protected_hijacked, f"Argument hijacking not prevented for {model_name}"


class TestOllama:
    """Argument hijacking tests for Ollama models."""

    @pytest.mark.requires("langchain_ollama")
    @pytest.mark.parametrize(
        (
            "user_request",
            "payload",
            "tools",
            "expected_tool",
            "legitimate_args",
            "malicious_patterns",
        ),
        ARG_HIJACK_TEST_CASES,
    )
    def test_arg_hijacking(
        self,
        ollama_model,
        user_request,
        payload,
        tools,
        expected_tool,
        legitimate_args,
        malicious_patterns,
    ):
        _, protected_hijacked, model_name = _run_arg_hijack_test(
            ollama_model,
            user_request,
            payload,
            tools,
            expected_tool,
            legitimate_args,
            malicious_patterns,
        )
        assert not protected_hijacked, f"Argument hijacking not prevented for {model_name}"
