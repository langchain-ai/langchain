"""Tool for the identification of prompt injection attacks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

from langchain.pydantic_v1 import Field, root_validator
from langchain.tools.base import BaseTool

if TYPE_CHECKING:
    from transformers import Pipeline


class PromptInjectionException(ValueError):
    """Exception raised when prompt injection attack is detected."""

    def __init__(
        self, message: str = "Prompt injection attack detected", score: float = 1.0
    ):
        self.message = message
        self.score = score

        super().__init__(self.message)


def _model_default_factory(
    model_name: str = "protectai/deberta-v3-base-prompt-injection-v2",
) -> Pipeline:
    try:
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            pipeline,
        )
    except ImportError as e:
        raise ImportError(
            "Cannot import transformers, please install with "
            "`pip install transformers`."
        ) from e

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    return pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        max_length=512,  # default length of BERT models
        truncation=True,  # otherwise it will fail on long prompts
    )


class HuggingFaceInjectionIdentifier(BaseTool):
    """Tool that uses HuggingFace Prompt Injection model to
    detect prompt injection attacks."""

    name: str = "hugging_face_injection_identifier"
    description: str = (
        "A wrapper around HuggingFace Prompt Injection security model. "
        "Useful for when you need to ensure that prompt is free of injection attacks. "
        "Input should be any message from the user."
    )
    model: Union[Pipeline, str, None] = Field(default_factory=_model_default_factory)
    """Model to use for prompt injection detection. 
    
    Can be specified as transformers Pipeline or string. String should correspond to the
        model name of a text-classification transformers model. Defaults to 
        ``protectai/deberta-v3-base-prompt-injection-v2`` model.
    """
    threshold: float = Field(
        description="Threshold for prompt injection detection.", default=0.5
    )
    """Threshold for prompt injection detection.
    
    Defaults to 0.5."""
    injection_label: str = Field(
        description="Label of the injection for prompt injection detection.",
        default="INJECTION",
    )
    """Label for prompt injection detection model.
    
    Defaults to ``INJECTION``. Value depends on the model used."""

    @root_validator(pre=True)
    def validate_environment(cls, values: dict) -> dict:
        if isinstance(values.get("model"), str):
            values["model"] = _model_default_factory(model_name=values["model"])
        return values

    def _run(self, query: str) -> str:
        """Use the tool."""
        result = self.model(query)  # type: ignore
        score = (
            result[0]["score"]
            if result[0]["label"] == self.injection_label
            else 1 - result[0]["score"]
        )
        if score > self.threshold:
            raise PromptInjectionException("Prompt injection attack detected", score)

        return query
