"""Tool for the identification of prompt injection attacks."""

from enum import Enum

from langchain.tools.base import BaseTool
from transformers import Pipeline, pipeline


class PromptInjectionModelOutput(str, Enum):
    """Output of the prompt injection model."""

    LEGIT = "LEGIT"
    INJECTION = "INJECTION"


class HuggingFaceInjectionIdentifier(BaseTool):
    """Tool that uses deberta-v3-base-injection model
    to identify prompt injection attacks."""

    name: str = "hugging_face_injection_identifier"
    description: str = (
        "A wrapper around HuggingFace Prompt Injection security model. "
        "Useful for when you need to ensure that prompt is free of injection attacks. "
        "Input should be any message from the user."
    )

    model: Pipeline = pipeline(
        "text-classification", model="deepset/deberta-v3-base-injection"
    )

    def _classify_user_input(self, query: str) -> bool:
        result = self.model(query)
        result = sorted(result, key=lambda x: x["score"], reverse=True)
        if result[0]["label"] == PromptInjectionModelOutput.INJECTION:
            return False
        return True

    def _run(self, query: str) -> str:
        """Use the tool."""
        is_query_safe = self._classify_user_input(query)
        if not is_query_safe:
            raise ValueError("Prompt injection attack detected")
        return query
