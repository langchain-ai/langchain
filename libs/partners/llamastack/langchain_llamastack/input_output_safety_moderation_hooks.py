"""
Input/Output Safety Hooks

Simple, efficient implementation with 2 essential hooks:
1. Input Hook - Check user input safety BEFORE sending to LLM
2. Output Hook - Check model output safety AFTER LLM generates response

Each hook uses LlamaStack's run_shield API once to get comprehensive safety results.
"""

from typing import Any, Callable, Optional

from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig

from .safety import SafetyResult


class SafeLLMWrapper(Runnable):
    """
    Simple LLM wrapper with input/output safety hooks.

    Flow: User Input → Input Hook → LLM → Output Hook → Safe Response
    """

    def __init__(self, llm: Any, safety_client: Any) -> None:
        self.llm = llm
        self.safety_client = safety_client
        self.input_hook: Optional[Callable[[str], SafetyResult]] = None
        self.output_hook: Optional[Callable[[str], SafetyResult]] = None

    def set_input_hook(self, hook: Callable[[str], SafetyResult]) -> None:
        """Set input hook (checks user input before LLM)."""
        self.input_hook = hook

    def set_output_hook(self, hook: Callable[[str], SafetyResult]) -> None:
        """Set output hook (checks model output after LLM)."""
        self.output_hook = hook

    def invoke(
        self, input: Any, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> str:
        """Run LLM with safety checks."""

        # Extract string input if needed
        if isinstance(input, dict):
            user_input = input.get("input") or input.get("question") or str(input)
        else:
            user_input = str(input)

        # 1. INPUT CHECK (safety using run_shield)
        if self.input_hook is not None:
            input_result = self.input_hook(user_input)
            if not input_result.is_safe:
                violations = [
                    v.get("reason", "Safety violation") for v in input_result.violations
                ]
                return f"Input blocked by safety system: {'; '.join(violations)}"

        # 2. LLM EXECUTION
        try:
            llm_response = self.llm.invoke(user_input)
            # Extract text from different LLM response formats
            if hasattr(llm_response, "content"):
                model_output = llm_response.content
            elif isinstance(llm_response, str):
                model_output = llm_response
            else:
                model_output = str(llm_response)
        except Exception as e:
            return f"LLM execution failed: {str(e)}"

        # 3. OUTPUT CHECK (safety using run_shield)
        if self.output_hook is not None:
            output_result = self.output_hook(model_output)
            if not output_result.is_safe:
                violations = [
                    v.get("reason", "Safety violation")
                    for v in output_result.violations
                ]
                return f"Output blocked by safety system: {'; '.join(violations)}"

        return model_output

    async def ainvoke(
        self, input: Any, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> str:
        """Async version of invoke."""

        # Extract string input if needed
        if isinstance(input, dict):
            user_input = input.get("input") or input.get("question") or str(input)
        else:
            user_input = str(input)

        # 1. INPUT CHECK (safety using run_shield)
        if self.input_hook is not None:
            input_result = self.input_hook(user_input)
            if not input_result.is_safe:
                violations = [
                    v.get("reason", "Safety violation") for v in input_result.violations
                ]
                return f"Input blocked by safety system: {'; '.join(violations)}"

        # 2. LLM EXECUTION
        try:
            llm_response = await self.llm.ainvoke(user_input)
            # Extract text from different LLM response formats
            if hasattr(llm_response, "content"):
                model_output = llm_response.content
            elif isinstance(llm_response, str):
                model_output = llm_response
            else:
                model_output = str(llm_response)
        except Exception as e:
            return f"LLM execution failed: {str(e)}"

        # 3. OUTPUT CHECK (safety using run_shield)
        if self.output_hook is not None:
            output_result = self.output_hook(model_output)
            if not output_result.is_safe:
                violations = [
                    v.get("reason", "Safety violation")
                    for v in output_result.violations
                ]
                return f"Output blocked by safety system: {'; '.join(violations)}"

        return model_output


# =============================================================================
# The 2 Essential Hooks Using LlamaStack run_shield API
# =============================================================================


def create_safety_hook(
    safety_client: Any, hook_type: str = "input"
) -> Callable[[str], SafetyResult]:
    """
    Create a safety hook using LlamaStack's run_shield API.

    Args:
        safety_client: LlamaStackSafety client instance
        hook_type: Type of hook - "input" (fails open) or "output" (fails closed)

    Returns:
        Function that takes content and returns SafetyResult
    """
    fail_open = hook_type == "input"

    def safety_hook(content: str) -> SafetyResult:
        try:
            return safety_client.check_content_safety(content)
        except Exception as e:
            if fail_open:
                # Input hooks fail open - allow content to proceed but log error
                return SafetyResult(
                    is_safe=True,
                    violations=[],
                    explanation=f"Safety check failed: {e}",
                )
            else:
                # Output hooks fail closed - block content on error
                return SafetyResult(
                    is_safe=False,
                    violations=[{"category": "check_error", "reason": str(e)}],
                    explanation=f"Safety check failed: {e}",
                )

    return safety_hook


# =============================================================================
# Convenient Factory Functions
# ======================================================================


def create_safe_llm(
    llm: Any, safety_client: Any, input_check: bool = True, output_check: bool = True
) -> SafeLLMWrapper:
    """
    Create a safe LLM with configurable input/output checking.

    This is the main factory function that provides clean safety hooks.
    Each check uses LlamaStack's run_shield once for comprehensive safety checking.

    Args:
        llm: The language model to wrap
        safety_client: LlamaStackSafety client instance
        input_check: Whether to check user input (default: True)
        output_check: Whether to check model output (default: True)

    Returns:
        SafeLLMWrapper with configured hooks

    Examples:
        # Complete protection (recommended)
        safe_llm = create_safe_llm(llm, safety_client)

        # Input filtering only
        safe_llm = create_safe_llm(llm, safety_client, output_check=False)

        # Output filtering only
        safe_llm = create_safe_llm(llm, safety_client, input_check=False)

        # No protection (same as unwrapped LLM)
        safe_llm = create_safe_llm(llm, safety_client,\
         input_check=False, output_check=False)
    """
    safe_llm = SafeLLMWrapper(llm, safety_client)

    # Set hooks based on configuration
    if input_check:
        safe_llm.set_input_hook(create_safety_hook(safety_client, "input"))

    if output_check:
        safe_llm.set_output_hook(create_safety_hook(safety_client, "output"))

    return safe_llm


def create_safe_llm_with_all_hooks(llm: Any, safety_client: Any) -> SafeLLMWrapper:
    """
    Create a safe LLM with complete protection (both input and output checking).

    This provides maximum safety by checking both user input and model output.
    Equivalent to: create_safe_llm(llm, safety_client, input_check=True,\
     output_check=True)

    Args:
        llm: The language model to wrap
        safety_client: LlamaStackSafety client instance

    Returns:
        SafeLLMWrapper with complete protection
    """
    return create_safe_llm(llm, safety_client, input_check=True, output_check=True)


def create_input_only_safe_llm(llm: Any, safety_client: Any) -> SafeLLMWrapper:
    """
    Create a safe LLM with only input checking.

    Use this when you trust the model outputs but want to filter user inputs.
    Equivalent to: create_safe_llm(llm, safety_client, output_check=False)

    Args:
        llm: The language model to wrap
        safety_client: LlamaStackSafety client instance

    Returns:
        SafeLLMWrapper with input checking only
    """
    return create_safe_llm(llm, safety_client, input_check=True, output_check=False)


def create_output_only_safe_llm(llm: Any, safety_client: Any) -> SafeLLMWrapper:
    """
    Create a safe LLM with only output checking.

    Use this when you trust user inputs but want to filter model outputs.
    Equivalent to: create_safe_llm(llm, safety_client, input_check=False)

    Args:
        llm: The language model to wrap
        safety_client: LlamaStackSafety client instance

    Returns:
        SafeLLMWrapper with output checking only
    """
    return create_safe_llm(llm, safety_client, input_check=False, output_check=True)
