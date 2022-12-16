"""Wrapper around HuggingFace APIs."""
from typing import Any, Dict, List, Mapping, Optional
from pydantic import BaseModel, Extra, root_validator
from langchain.llms.base import LLM


DEFAULT_MODEL_ID = "gpt2"
VALID_TASKS = ("text2text-generation", "text-generation")


class HuggingFaceLocal(LLM, BaseModel):
    pipeline: Any  #: :meta private:
    model_id: str = DEFAULT_MODEL_ID
    """Model name to use."""
    task: Optional[str] = None
    """Task to call the model with. Should be a task that returns `generated_text`."""
    model_kwargs: Optional[dict] = None
    """Key word arguments to pass to the model."""

    class Config:
        """Configuration for this pydantic object."""
        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            model = AutoModelForCausalLM.from_pretrained("gpt2", problem_type="")
            local_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
            if local_pipeline.task not in VALID_TASKS:
                raise ValueError(
                    f"Got invalid task {local_pipeline.task}, "
                    f"currently only {VALID_TASKS} are supported"
                )
        except ImportError:
            raise ValueError(
                "Could not import transformers python package. "
                "Please it install it with `pip install transformers`."
            )
        return values

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        _model_kwargs = self.model_kwargs or {}
        return {
            **{"repo_id": self.repo_id, "task": self.task},
            **{"model_kwargs": _model_kwargs},
        }

    @property
    def _llm_type(self) -> str:
        return "huggingface_local"

    def __call__(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        _model_kwargs = self.model_kwargs or {}
        response = self.pipeline(inputs=prompt, params=_model_kwargs)
        if "error" in response:
            raise ValueError(f"Error raised by inference API: {response['error']}")
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
        return text
