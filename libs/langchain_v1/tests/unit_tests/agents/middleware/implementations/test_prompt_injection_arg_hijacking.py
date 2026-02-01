"""Argument hijacking prompt injection tests.

Tests a more subtle attack vector: the user legitimately wants to call a tool,
but the injection manipulates the arguments (e.g., changing email recipient,
appending malicious content to body, swapping URLs).

This is harder to detect than unauthorized tool triggering because:
1. The tool call itself is expected
2. Only the arguments are compromised
3. The attack blends with legitimate data

Compares effectiveness of different defense strategies:
1. CombinedStrategy (CheckTool + ParseData) - existing approach
2. IntentVerificationStrategy - new intent-based approach
3. Both combined - all three strategies together

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

from langchain.agents.middleware import (
    CheckToolStrategy,
    CombinedStrategy,
    IntentVerificationStrategy,
    ParseDataStrategy,
)

from .conftest import (
    ARG_HIJACK_TEST_CASES,
    check_argument_hijacking,
    create_tool_message,
    create_tool_request_with_user_message,
)


def _get_model_name(model):
    """Extract model name from different model types."""
    return getattr(model, "model_name", None) or getattr(model, "model", "unknown")


def _run_arg_hijack_test(
    model, user_request, payload, tools, expected_tool, legitimate_args, malicious_patterns
):
    """Test argument hijacking with different defense strategies.

    Returns:
        dict with results for each strategy
    """
    model_name = _get_model_name(model)

    # Baseline: check if model is vulnerable without protection
    baseline_hijacked, _, baseline_details = check_argument_hijacking(
        model, tools, user_request, payload, expected_tool, legitimate_args, malicious_patterns
    )

    results = {
        "model": model_name,
        "baseline": {"hijacked": baseline_hijacked, "details": baseline_details},
    }

    # Strategy 1: CombinedStrategy (CheckTool + ParseData)
    combined_strategy = CombinedStrategy(
        [
            CheckToolStrategy(model, tools=tools),
            ParseDataStrategy(model, use_full_conversation=True),
        ]
    )
    req = create_tool_request_with_user_message(tools, "get_user_data", user_request)
    combined_result = combined_strategy.process(req, create_tool_message(payload, "get_user_data"))
    combined_hijacked, _, combined_details = check_argument_hijacking(
        model,
        tools,
        user_request,
        str(combined_result.content),
        expected_tool,
        legitimate_args,
        malicious_patterns,
    )
    results["combined"] = {
        "hijacked": combined_hijacked,
        "details": combined_details,
        "safe": not combined_hijacked,
    }

    # Strategy 2: IntentVerificationStrategy only
    intent_strategy = IntentVerificationStrategy(model)
    req = create_tool_request_with_user_message(tools, "get_user_data", user_request)
    intent_result = intent_strategy.process(req, create_tool_message(payload, "get_user_data"))
    intent_hijacked, _, intent_details = check_argument_hijacking(
        model,
        tools,
        user_request,
        str(intent_result.content),
        expected_tool,
        legitimate_args,
        malicious_patterns,
    )
    results["intent_only"] = {
        "hijacked": intent_hijacked,
        "details": intent_details,
        "safe": not intent_hijacked,
    }

    # Strategy 3: All strategies combined
    all_strategy = CombinedStrategy(
        [
            CheckToolStrategy(model, tools=tools),
            ParseDataStrategy(model, use_full_conversation=True),
            IntentVerificationStrategy(model),
        ]
    )
    req = create_tool_request_with_user_message(tools, "get_user_data", user_request)
    all_result = all_strategy.process(req, create_tool_message(payload, "get_user_data"))
    all_hijacked, _, all_details = check_argument_hijacking(
        model,
        tools,
        user_request,
        str(all_result.content),
        expected_tool,
        legitimate_args,
        malicious_patterns,
    )
    results["all_combined"] = {
        "hijacked": all_hijacked,
        "details": all_details,
        "safe": not all_hijacked,
    }

    # Print comparison
    baseline_str = "HIJACKED" if baseline_hijacked else "clean"
    combined_str = "SAFE" if results["combined"]["safe"] else "HIJACKED"
    intent_str = "SAFE" if results["intent_only"]["safe"] else "HIJACKED"
    all_str = "SAFE" if results["all_combined"]["safe"] else "HIJACKED"

    print(f"\n{model_name}:")
    print(f"  baseline={baseline_str} ({baseline_details})")
    print(f"  combined(CheckTool+ParseData)={combined_str} ({combined_details})")
    print(f"  intent_only={intent_str} ({intent_details})")
    print(f"  all_combined={all_str} ({all_details})")

    return results


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
        results = _run_arg_hijack_test(
            openai_model,
            user_request,
            payload,
            tools,
            expected_tool,
            legitimate_args,
            malicious_patterns,
        )
        # At least one strategy should protect
        assert (
            results["combined"]["safe"]
            or results["intent_only"]["safe"]
            or results["all_combined"]["safe"]
        ), f"No strategy prevented argument hijacking for {results['model']}"


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
        results = _run_arg_hijack_test(
            anthropic_model,
            user_request,
            payload,
            tools,
            expected_tool,
            legitimate_args,
            malicious_patterns,
        )
        assert (
            results["combined"]["safe"]
            or results["intent_only"]["safe"]
            or results["all_combined"]["safe"]
        ), f"No strategy prevented argument hijacking for {results['model']}"


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
        results = _run_arg_hijack_test(
            google_model,
            user_request,
            payload,
            tools,
            expected_tool,
            legitimate_args,
            malicious_patterns,
        )
        assert (
            results["combined"]["safe"]
            or results["intent_only"]["safe"]
            or results["all_combined"]["safe"]
        ), f"No strategy prevented argument hijacking for {results['model']}"


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
        results = _run_arg_hijack_test(
            ollama_model,
            user_request,
            payload,
            tools,
            expected_tool,
            legitimate_args,
            malicious_patterns,
        )
        assert (
            results["combined"]["safe"]
            or results["intent_only"]["safe"]
            or results["all_combined"]["safe"]
        ), f"No strategy prevented argument hijacking for {results['model']}"
