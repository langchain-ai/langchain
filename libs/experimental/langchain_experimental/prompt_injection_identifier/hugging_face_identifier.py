"""Tool for the identification of prompt injection attacks."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain.pydantic_v1 import Field, root_validator
from langchain.tools.base import BaseTool

if TYPE_CHECKING:
    from transformers import Pipeline


def _model_default_factory(
    model_name: str = "deepset/deberta-v3-base-injection"
) -> Pipeline:
    try:
        from transformers import pipeline
    except ImportError as e:
        raise ImportError(
            "Cannot import transformers, please install with "
            "`pip install transformers`."
        ) from e
    return pipeline("text-classification", model=model_name)


class HuggingFaceInjectionIdentifier(BaseTool):
    """Tool that uses HF model to detect prompt injection attacks."""

    name: str = "hugging_face_injection_identifier"
    description: str = (
        "A wrapper around HuggingFace Prompt Injection security model. "
        "Useful for when you need to ensure that prompt is free of injection attacks. "
        "Input should be any message from the user."
    )
    model: Any = Field(default_factory=_model_default_factory)
    """Model to use for prompt injection detection. 
    
    Can be specified as transformers Pipeline or string. String should correspond to the
        model name of a text-classification transformers model. Defaults to 
        ``deepset/deberta-v3-base-injection`` model.
    """

    @root_validator(pre=True)
    def validate_environment(cls, values: dict) -> dict:
        if isinstance(values.get("model"), str):
            values["model"] = _model_default_factory(model_name=values["model"])
        return values

    def _run(self, query: str) -> str:
        """Use the tool."""
        result = self.model(query)
        result = sorted(result, key=lambda x: x["score"], reverse=True)
        if result[0]["label"] == "INJECTION":
            raise ValueError("Prompt injection attack detected")
        return query
