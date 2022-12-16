"""Wrapper around HuggingFace Pipeline APIs."""
from typing import Any, Dict, List, Mapping, Optional

from pydantic import BaseModel, Extra, root_validator

from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens

DEFAULT_MODEL_ID = "gpt2"
DEFAULT_TASK = "text-generation"
VALID_TASKS = ("text2text-generation", "text-generation")


class HuggingFacePipeline(LLM, BaseModel):
    """Wrapper around HuggingFace Pipeline API.

    To use, you should have the ``transformers`` python package installed.

    Only supports `text-generation` and `text2text-generation` for now.

    Example:
        .. code-block:: python

            from langchain.llms.huggingface_pipeline import HuggingFacePipeline
            hf = HuggingFacePipeline(model_id="gpt2", task="text-generation")
    """

    pipeline: Any  #: :meta private:
    model_id: str = DEFAULT_MODEL_ID
    """Model name to use."""
    task: str = DEFAULT_TASK
    """Task to call the model with. Should be a task that returns `generated_text`."""
    model_kwargs: Optional[dict] = None
    """Key word arguments to pass to the model."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate python package exists in environment."""
        try:
            if values.get("pipeline") is None:
                values["pipeline"] = cls.from_model_id(values)

        except ImportError:
            raise ValueError(
                "Could not import transformers python package. "
                "Please it install it with `pip install transformers`."
            )

        return values

    @staticmethod
    def from_model_id(values):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers import pipeline as hf_pipeline

        model_id = values["model_id"]
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
        pipeline = hf_pipeline(
            values["task"],
            model=model,
            tokenizer=tokenizer,
            model_kwargs=values["model_kwargs"],
        )
        if pipeline.task not in VALID_TASKS:
            raise ValueError(
                f"Got invalid task {pipeline.task}, "
                f"currently only {VALID_TASKS} are supported"
            )
        return pipeline

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        _model_kwargs = self.model_kwargs or {}
        return {
            **{"model_id": self.model_id, "task": self.task},
            **{"model_kwargs": _model_kwargs},
        }

    @property
    def _llm_type(self) -> str:
        return "huggingface_pipeline"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self.pipeline(text_inputs=prompt)
        if self.pipeline.task == "text-generation":
            # Text generation return includes the starter text.
            text = response[0]["generated_text"][len(prompt) :]
        elif self.pipeline.task == "text2text-generation":
            text = response[0]["generated_text"]
        else:
            raise ValueError(
                f"Got invalid task {self.pipeline.task}, "
                f"currently only {VALID_TASKS} are supported"
            )
        if stop is not None:
            # This is a bit hacky, but I can't figure out a better way to enforce
            # stop tokens when making calls to huggingface_hub.
            text = enforce_stop_tokens(text, stop)
        return text
