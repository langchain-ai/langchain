import logging
from typing import Any, Callable, Dict, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.pydantic_v1 import root_validator

logger = logging.getLogger(__name__)


def default_guardrail_violation_handler(violation: dict) -> str:
    """Default guardrail violation handler.

    Args:
        violation (dict): The violation dictionary.

    Returns:
        str: The canned response.
    """
    if violation.get("canned_response"):
        return violation["canned_response"]
    guardrail_name = (
        f"Guardrail {violation.get('offending_guardrail')}"
        if violation.get("offending_guardrail")
        else "A guardrail"
    )
    raise ValueError(
        f"{guardrail_name} was violated without a proper guardrail violation handler."
    )


class LayerupSecurity(LLM):
    """Layerup Security LLM service."""

    llm: LLM
    layerup_api_key: str
    layerup_api_base_url: str = "https://api.uselayerup.com/v1"
    prompt_guardrails: Optional[List[str]] = []
    response_guardrails: Optional[List[str]] = []
    mask: bool = False
    metadata: Optional[Dict[str, Any]] = {}
    handle_prompt_guardrail_violation: Callable[[dict], str] = (
        default_guardrail_violation_handler
    )
    handle_response_guardrail_violation: Callable[[dict], str] = (
        default_guardrail_violation_handler
    )
    client: Any  #: :meta private:

    @root_validator(pre=True)
    def validate_layerup_sdk(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        try:
            from layerup_security import LayerupSecurity as LayerupSecuritySDK

            values["client"] = LayerupSecuritySDK(
                api_key=values["layerup_api_key"],
                base_url=values["layerup_api_base_url"],
            )
        except ImportError:
            raise ImportError(
                "Could not import LayerupSecurity SDK. "
                "Please install it with `pip install LayerupSecurity`."
            )
        return values

    @property
    def _llm_type(self) -> str:
        return "layerup_security"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        messages = [{"role": "user", "content": prompt}]
        unmask_response = None

        if self.mask:
            messages, unmask_response = self.client.mask_prompt(messages, self.metadata)

        if self.prompt_guardrails:
            security_response = self.client.execute_guardrails(
                self.prompt_guardrails, messages, prompt, self.metadata
            )
            if not security_response["all_safe"]:
                return self.handle_prompt_guardrail_violation(security_response)

        result = self.llm._call(
            messages[0]["content"], run_manager=run_manager, **kwargs
        )

        if self.mask and unmask_response:
            result = unmask_response(result)

        messages.append({"role": "assistant", "content": result})

        if self.response_guardrails:
            security_response = self.client.execute_guardrails(
                self.response_guardrails, messages, result, self.metadata
            )
            if not security_response["all_safe"]:
                return self.handle_response_guardrail_violation(security_response)

        return result
