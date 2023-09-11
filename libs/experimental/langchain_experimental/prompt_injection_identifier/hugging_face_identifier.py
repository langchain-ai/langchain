"""Tool for the identification of prompt injection attacks."""
from __future__ import annotations

from typing import TYPE_CHECKING

from langchain.pydantic_v1 import Field
from langchain.tools.base import BaseTool

if TYPE_CHECKING:
    from transformers import Pipeline


def _model_default_factory() -> Pipeline:
    try:
        from transformers import pipeline
    except ImportError as e:
        raise ImportError(
            "Cannot import transformers, please install with "
            "`pip install transformers`."
        ) from e
    return pipeline("text-classification", model="deepset/deberta-v3-base-injection")


class HuggingFaceInjectionIdentifier(BaseTool):
    """Tool that uses deberta-v3-base-injection to detect prompt injection attacks."""

    name: str = "hugging_face_injection_identifier"
    description: str = (
        "A wrapper around HuggingFace Prompt Injection security model. "
        "Useful for when you need to ensure that prompt is free of injection attacks. "
        "Input should be any message from the user."
    )
    model: Pipeline = Field(default_factory=_model_default_factory)

    def _run(self, query: str) -> str:
        """Use the tool."""
        result = self.model(query)
        result = sorted(result, key=lambda x: x["score"], reverse=True)
        if result[0]["label"] == "INJECTION":
            raise ValueError("Prompt injection attack detected")
        return query
