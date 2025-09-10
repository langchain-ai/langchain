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
    Universal safety wrapper for LLMs and Agents.

    Supports:
    - Language Models (ChatOpenAI, etc.)
    - LangChain Agents (LangGraph, AgentExecutor, etc.)
    - Any Runnable or callable object

    Flow: User Input → Input Hook → LLM/Agent → Output Hook → Safe Response
    """

    def __init__(self, runnable: Any, safety_client: Any) -> None:
        self.runnable = runnable  # Can be LLM, Agent, or any Runnable
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

        # 2. RUNNABLE EXECUTION (LLM, Agent, etc.)
        try:
            # Handle different execution methods
            if hasattr(self.runnable, "invoke"):
                response = self.runnable.invoke(user_input)
            elif hasattr(self.runnable, "run"):
                # Legacy AgentExecutor
                response = self.runnable.run(user_input)
            elif callable(self.runnable):
                response = self.runnable(user_input)
            else:
                raise ValueError(f"Unsupported runnable type: {type(self.runnable)}")

            # Extract text from different response formats
            if hasattr(response, "content"):
                model_output = response.content
            elif isinstance(response, dict):
                # Agent responses might be dicts
                model_output = (
                    response.get("output") or response.get("answer") or str(response)
                )
            elif isinstance(response, str):
                model_output = response
            else:
                model_output = str(response)
        except Exception as e:
            return f"Execution failed: {str(e)}"

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

        # 2. ASYNC RUNNABLE EXECUTION (LLM, Agent, etc.)
        try:
            # Handle different async execution methods
            if hasattr(self.runnable, "ainvoke"):
                response = await self.runnable.ainvoke(user_input)
            elif hasattr(self.runnable, "arun"):
                # Legacy async AgentExecutor
                response = await self.runnable.arun(user_input)
            else:
                # Fallback to sync execution
                response = self.runnable.invoke(user_input)

            # Extract text from different response formats
            if hasattr(response, "content"):
                model_output = response.content
            elif isinstance(response, dict):
                # Agent responses might be dicts
                model_output = (
                    response.get("output") or response.get("answer") or str(response)
                )
            elif isinstance(response, str):
                model_output = response
            else:
                model_output = str(response)
        except Exception as e:
            return f"Async execution failed: {str(e)}"

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
    safety_client: Any, hook_type: str = "input", shield_type: Optional[str] = None
) -> Callable[[str], SafetyResult]:
    """
    Create a safety hook using LlamaStack's run_shield API.

    Args:
        safety_client: LlamaStackSafety client instance
        hook_type: Type of hook - "input" (fails open) or "output" (fails closed)
        shield_type: Specific shield to use. If None, uses optimal defaults:
                    - "prompt_guard" for input hooks (prompt injection detection)
                    - "llama_guard" for output hooks (content moderation)

    Returns:
        Function that takes content and returns SafetyResult
    """
    fail_open = hook_type == "input"

    # Use optimal shield defaults based on hook type
    if shield_type is None:
        shield_type = "prompt_guard" if hook_type == "input" else "llama_guard"

    def safety_hook(content: str) -> SafetyResult:
        try:
            # Create a temporary safety client with the specific shield type if different
            if shield_type != getattr(safety_client, "shield_type", "llama_guard"):
                from .safety import LlamaStackSafety

                temp_client = LlamaStackSafety(
                    base_url=safety_client.base_url,
                    shield_type=shield_type,
                    timeout=safety_client.timeout,
                    max_retries=safety_client.max_retries,
                )
                return temp_client.check_content_safety(content)
            else:
                # Use the provided safety client if shield type matches
                return safety_client.check_content_safety(content)
        except Exception as e:
            if fail_open:
                # Input hooks fail open - allow content to proceed but log error
                return SafetyResult(
                    is_safe=True,
                    violations=[],
                    explanation=f"Safety check failed with {shield_type}: {e}",
                )
            else:
                # Output hooks fail closed - block content on error
                return SafetyResult(
                    is_safe=False,
                    violations=[{"category": "check_error", "reason": str(e)}],
                    explanation=f"Safety check failed with {shield_type}: {e}",
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
        safe_llm = create_safe_llm(llm, safety_client, input_check=False, output_check=False)
    """
    safe_llm = SafeLLMWrapper(llm, safety_client)

    # Set hooks based on configuration
    if input_check:
        safe_llm.set_input_hook(create_safety_hook(safety_client, "input"))

    if output_check:
        safe_llm.set_output_hook(
            create_safety_hook(
                safety_client, "output", shield_type=safety_client.shield_type
            )
        )

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
