# flake8: noqa
from typing import Any, Dict, Optional, List

from langchain.pydantic_v1 import root_validator

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens


class DeepSparse(LLM):
    """Neural Magic DeepSparse LLM interface.

    To use, you should have the ``deepsparse`` or ``deepsparse-nightly``
    python package installed. See https://github.com/neuralmagic/deepsparse

    This interface let's you deploy optimized LLMs straight from the
    [SparseZoo](https://sparsezoo.neuralmagic.com/?useCase=text_generation)
    Example:
        .. code-block:: python
            from langchain.llms import DeepSparse
            llm = DeepSparse(model="zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/bigpython_bigquery_thepile/base-none")
    """  # noqa: E501

    pipeline: Any  #: :meta private:

    model: str
    """The path to a model file or directory or the name of a SparseZoo model stub."""

    config: Optional[Dict[str, Any]] = None
    """Key word arguments passed to the pipeline."""

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "model": self.model,
            "config": self.config,
        }

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "deepsparse"

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that ``deepsparse`` package is installed."""
        try:
            from deepsparse import Pipeline
        except ImportError:
            raise ImportError(
                "Could not import `deepsparse` package. "
                "Please install it with `pip install deepsparse`"
            )

        config = values["config"] or {}

        values["pipeline"] = Pipeline.create(
            task="text_generation",
            model_path=values["model"],
            **config,
        )
        return values

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Generate text from a prompt.
        Args:
            prompt: The prompt to generate text from.
            stop: A list of strings to stop generation when encountered.
        Returns:
            The generated text.
        Example:
            .. code-block:: python
                response = llm("Tell me a joke.")
        """
        text = self.pipeline(sequences=prompt).sequences[0]

        if stop is not None:
            text = enforce_stop_tokens(text, stop)
        return "".join(text)
