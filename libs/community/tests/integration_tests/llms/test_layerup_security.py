from typing import Any, List, Optional

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM

from langchain_community.llms.layerup_security import LayerupSecurity


class MockLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "mock_llm"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        return "Hi Bob! How are you?"


def test_layerup_security_with_invalid_api_key() -> None:
    mock_llm = MockLLM()
    layerup_security = LayerupSecurity(
        llm=mock_llm,
        layerup_api_key="-- invalid API key --",
        layerup_api_base_url="https://api.uselayerup.com/v1",
        prompt_guardrails=[],
        response_guardrails=["layerup.hallucination"],
        mask=False,
        metadata={"customer": "example@uselayerup.com"},
        handle_response_guardrail_violation=(
            lambda violation: (
                "Custom canned response with dynamic data! "
                "The violation rule was {offending_guardrail}."
            ).format(offending_guardrail=violation["offending_guardrail"])
        ),
    )

    with pytest.raises(Exception):
        layerup_security.invoke("My name is Bob Dylan. My SSN is 123-45-6789.")
