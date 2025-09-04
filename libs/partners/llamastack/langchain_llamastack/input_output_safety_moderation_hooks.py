"""
Input/Output Safety & Moderation Hooks

Simple, focused implementation with 4 essential hooks:
1. Input Safety Hook - Check user input safety BEFORE sending to LLM
2. Input Moderation Hook - Moderate user input BEFORE sending to LLM
3. Output Safety Hook - Check model output safety AFTER LLM generates response
4. Output Moderation Hook - Moderate model output AFTER LLM generates response

All hooks use LlamaStack's run_shield API for accurate, context-aware checking.
"""

from typing import Callable

from langchain_core.runnables import Runnable

from .safety import SafetyResult


class SafeLLMWrapper(Runnable):
    """
    Simple LLM wrapper with input/output safety and moderation hooks.

    Flow: User Input → Input Hooks → LLM → Output Hooks → Safe Response
    """

    def __init__(self, llm, safety_client):
        self.llm = llm
        self.safety_client = safety_client
        self.input_safety_hook: Callable[[str], SafetyResult] = None
        self.input_moderation_hook: Callable[[str], SafetyResult] = None
        self.output_safety_hook: Callable[[str], SafetyResult] = None
        self.output_moderation_hook: Callable[[str], SafetyResult] = None

    def set_input_safety_hook(self, hook: Callable[[str], SafetyResult]) -> None:
        """Set input safety hook (checks user input before LLM)."""
        self.input_safety_hook = hook

    def set_input_moderation_hook(self, hook: Callable[[str], SafetyResult]) -> None:
        """Set input moderation hook (moderates user input before LLM)."""
        self.input_moderation_hook = hook

    def set_output_safety_hook(self, hook: Callable[[str], SafetyResult]) -> None:
        """Set output safety hook (checks model output after LLM)."""
        self.output_safety_hook = hook

    def set_output_moderation_hook(self, hook: Callable[[str], SafetyResult]) -> None:
        """Set output moderation hook (moderates model output after LLM)."""
        self.output_moderation_hook = hook

    def invoke(self, input_text: str, config=None) -> str:
        """Run LLM with safety and moderation checks."""

        # Extract string input if needed
        if isinstance(input_text, dict):
            user_input = (
                input_text.get("input") or input_text.get("question") or str(input_text)
            )
        else:
            user_input = str(input_text)

        # 1. INPUT SAFETY CHECK
        if self.input_safety_hook:
            input_safety_result = self.input_safety_hook(user_input)
            if not input_safety_result.is_safe:
                violations = [
                    v.get("reason", "Safety violation")
                    for v in input_safety_result.violations
                ]
                return f"Input blocked by safety system: {'; '.join(violations)}"

        # 2. INPUT MODERATION CHECK
        if self.input_moderation_hook:
            input_moderation_result = self.input_moderation_hook(user_input)
            if not input_moderation_result.is_safe:
                violations = [
                    v.get("reason", "Moderation violation")
                    for v in input_moderation_result.violations
                ]
                return f"Input blocked by moderation system: {'; '.join(violations)}"

        # 3. LLM EXECUTION
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

        # 4. OUTPUT SAFETY CHECK
        if self.output_safety_hook:
            output_safety_result = self.output_safety_hook(model_output)
            if not output_safety_result.is_safe:
                violations = [
                    v.get("reason", "Safety violation")
                    for v in output_safety_result.violations
                ]
                return f"Output blocked by safety system: {'; '.join(violations)}"

        # 5. OUTPUT MODERATION CHECK
        if self.output_moderation_hook:
            output_moderation_result = self.output_moderation_hook(model_output)
            if not output_moderation_result.is_safe:
                violations = [
                    v.get("reason", "Moderation violation")
                    for v in output_moderation_result.violations
                ]
                return f"Output blocked by moderation system: {'; '.join(violations)}"

        return model_output

    async def ainvoke(self, input_text: str, config=None) -> str:
        """Async version of invoke."""

        # Extract string input if needed
        if isinstance(input_text, dict):
            user_input = (
                input_text.get("input") or input_text.get("question") or str(input_text)
            )
        else:
            user_input = str(input_text)

        # 1. INPUT SAFETY CHECK
        if self.input_safety_hook:
            input_safety_result = self.input_safety_hook(user_input)
            if not input_safety_result.is_safe:
                violations = [
                    v.get("reason", "Safety violation")
                    for v in input_safety_result.violations
                ]
                return f"Input blocked by safety system: {'; '.join(violations)}"

        # 2. INPUT MODERATION CHECK
        if self.input_moderation_hook:
            input_moderation_result = self.input_moderation_hook(user_input)
            if not input_moderation_result.is_safe:
                violations = [
                    v.get("reason", "Moderation violation")
                    for v in input_moderation_result.violations
                ]
                return f"Input blocked by moderation system: {'; '.join(violations)}"

        # 3. LLM EXECUTION
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

        # 4. OUTPUT SAFETY CHECK
        if self.output_safety_hook:
            output_safety_result = self.output_safety_hook(model_output)
            if not output_safety_result.is_safe:
                violations = [
                    v.get("reason", "Safety violation")
                    for v in output_safety_result.violations
                ]
                return f"Output blocked by safety system: {'; '.join(violations)}"

        # 5. OUTPUT MODERATION CHECK
        if self.output_moderation_hook:
            output_moderation_result = self.output_moderation_hook(model_output)
            if not output_moderation_result.is_safe:
                violations = [
                    v.get("reason", "Moderation violation")
                    for v in output_moderation_result.violations
                ]
                return f"Output blocked by moderation system: {'; '.join(violations)}"

        return model_output


# =============================================================================
# The 4 Essential Hooks Using LlamaStack run_shield API
# =============================================================================


def create_input_safety_hook(safety_client) -> Callable[[str], SafetyResult]:
    """
    Create input safety hook using LlamaStack's run_shield API.

    Checks user input for safety violations BEFORE sending to LLM.
    Uses LlamaStack's safety shields for accurate, context-aware checking.

    Args:
        safety_client: LlamaStackSafety client instance

    Returns:
        Function that takes user input and returns SafetyResult
    """

    def check_input_safety(user_input: str) -> SafetyResult:
        """Check user input safety using LlamaStack shields."""
        try:
            # Use LlamaStack's check_content_safety which calls run_shield internally
            return safety_client.check_content_safety(user_input)
        except Exception as e:
            # Fail open for input safety - allow user to proceed but log error
            return SafetyResult(
                is_safe=True,
                violations=[],
                explanation=f"Input safety check failed: {e}",
            )

    return check_input_safety


def create_input_moderation_hook(safety_client) -> Callable[[str], SafetyResult]:
    """
    Create input moderation hook using LlamaStack's moderation API.

    Moderates user input for policy violations BEFORE sending to LLM.
    Uses LlamaStack's moderation system for content policy enforcement.

    Args:
        safety_client: LlamaStackSafety client instance

    Returns:
        Function that takes user input and returns SafetyResult
    """

    def moderate_input(user_input: str) -> SafetyResult:
        """Moderate user input using LlamaStack moderation."""
        try:
            # Use LlamaStack's moderate_content for policy enforcement
            return safety_client.moderate_content(user_input)
        except Exception as e:
            # Fail open for input moderation - allow user to proceed but log error
            return SafetyResult(
                is_safe=True,
                violations=[],
                explanation=f"Input moderation failed: {e}",
            )

    return moderate_input


def create_output_safety_hook(safety_client) -> Callable[[str], SafetyResult]:
    """
    Create output safety hook using LlamaStack's run_shield API.

    Checks model output for safety violations AFTER LLM generates response.
    Uses LlamaStack's safety shields to prevent harmful model outputs.

    Args:
        safety_client: LlamaStackSafety client instance

    Returns:
        Function that takes model output and returns SafetyResult
    """

    def check_output_safety(model_output: str) -> SafetyResult:
        """Check model output safety using LlamaStack shields."""
        try:
            # Use LlamaStack's check_content_safety which calls run_shield internally
            return safety_client.check_content_safety(model_output)
        except Exception as e:
            # Fail closed for output safety - block output on errors
            return SafetyResult(
                is_safe=False,
                violations=[{"category": "safety_check_error", "reason": str(e)}],
                explanation=f"Output safety check failed: {e}",
            )

    return check_output_safety


def create_output_moderation_hook(safety_client) -> Callable[[str], SafetyResult]:
    """
    Create output moderation hook using LlamaStack's moderation API.

    Moderates model output for policy violations AFTER LLM generates response.
    Uses LlamaStack's moderation system to ensure compliant model outputs.

    Args:
        safety_client: LlamaStackSafety client instance

    Returns:
        Function that takes model output and returns SafetyResult
    """

    def moderate_output(model_output: str) -> SafetyResult:
        """Moderate model output using LlamaStack moderation."""
        try:
            # Use LlamaStack's moderate_content for policy enforcement
            return safety_client.moderate_content(model_output)
        except Exception as e:
            # Fail closed for output moderation - block output on errors
            return SafetyResult(
                is_safe=False,
                violations=[{"category": "moderation_error", "reason": str(e)}],
                explanation=f"Output moderation failed: {e}",
            )

    return moderate_output


# =============================================================================
# Convenient Factory Functions
# =============================================================================


def create_safe_llm_with_all_hooks(llm, safety_client) -> SafeLLMWrapper:
    """
    Create a safe LLM with all 4 hooks enabled.

    This provides complete protection:
    - Input safety + moderation (before LLM)
    - Output safety + moderation (after LLM)

    Args:
        llm: The language model to wrap
        safety_client: LlamaStackSafety client instance

    Returns:
        SafeLLMWrapper with all hooks configured
    """

    safe_llm = SafeLLMWrapper(llm, safety_client)

    # Set all 4 hooks
    safe_llm.set_input_safety_hook(create_input_safety_hook(safety_client))
    safe_llm.set_input_moderation_hook(create_input_moderation_hook(safety_client))
    safe_llm.set_output_safety_hook(create_output_safety_hook(safety_client))
    safe_llm.set_output_moderation_hook(create_output_moderation_hook(safety_client))

    return safe_llm


def create_input_only_safe_llm(llm, safety_client) -> SafeLLMWrapper:
    """
    Create a safe LLM with only input hooks (safety + moderation).

    Use this when you trust the model outputs but want to filter inputs.

    Args:
        llm: The language model to wrap
        safety_client: LlamaStackSafety client instance

    Returns:
        SafeLLMWrapper with input hooks only
    """

    safe_llm = SafeLLMWrapper(llm, safety_client)

    # Set only input hooks
    safe_llm.set_input_safety_hook(create_input_safety_hook(safety_client))
    safe_llm.set_input_moderation_hook(create_input_moderation_hook(safety_client))

    return safe_llm


def create_output_only_safe_llm(llm, safety_client) -> SafeLLMWrapper:
    """
    Create a safe LLM with only output hooks (safety + moderation).

    Use this when you trust user inputs but want to filter model outputs.

    Args:
        llm: The language model to wrap
        safety_client: LlamaStackSafety client instance

    Returns:
        SafeLLMWrapper with output hooks only
    """

    safe_llm = SafeLLMWrapper(llm, safety_client)

    # Set only output hooks
    safe_llm.set_output_safety_hook(create_output_safety_hook(safety_client))
    safe_llm.set_output_moderation_hook(create_output_moderation_hook(safety_client))

    return safe_llm


def create_safety_only_llm(llm, safety_client) -> SafeLLMWrapper:
    """
    Create a safe LLM with only safety hooks (input + output).

    Use this when you want safety checking but not content moderation.

    Args:
        llm: The language model to wrap
        safety_client: LlamaStackSafety client instance

    Returns:
        SafeLLMWrapper with safety hooks only
    """

    safe_llm = SafeLLMWrapper(llm, safety_client)

    # Set only safety hooks
    safe_llm.set_input_safety_hook(create_input_safety_hook(safety_client))
    safe_llm.set_output_safety_hook(create_output_safety_hook(safety_client))

    return safe_llm


def create_moderation_only_llm(llm, safety_client) -> SafeLLMWrapper:
    """
    Create a safe LLM with only moderation hooks (input + output).

    Use this when you want content moderation but not safety checking.

    Args:
        llm: The language model to wrap
        safety_client: LlamaStackSafety client instance

    Returns:
        SafeLLMWrapper with moderation hooks only
    """

    safe_llm = SafeLLMWrapper(llm, safety_client)

    # Set only moderation hooks
    safe_llm.set_input_moderation_hook(create_input_moderation_hook(safety_client))
    safe_llm.set_output_moderation_hook(create_output_moderation_hook(safety_client))

    return safe_llm
