"""Wrapper around Google VertexAI models."""
import asyncio
from concurrent.futures import Executor, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional

from pydantic import BaseModel, root_validator

from langchain.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from langchain.utilities.vertexai import (
    init_vertexai,
    raise_vertex_import_error,
)

if TYPE_CHECKING:
    from vertexai.language_models._language_models import _LanguageModel


def is_codey_model(model_name: str) -> bool:
    return "code" in model_name


class _VertexAICommon(BaseModel):
    client: "_LanguageModel" = None  #: :meta private:
    model_name: str
    "Model name to use."
    temperature: float = 0.0
    "Sampling temperature, it controls the degree of randomness in token selection."
    max_output_tokens: int = 128
    "Token limit determines the maximum amount of text output from one prompt."
    top_p: float = 0.95
    "Tokens are selected from most probable to least until the sum of their "
    "probabilities equals the top-p value. Top-p is ignored for Codey models."
    top_k: int = 40
    "How the model selects tokens for output, the next token is selected from "
    "among the top-k most probable tokens. Top-k is ignored for Codey models."
    stop: Optional[List[str]] = None
    "Optional list of stop words to use when generating."
    project: Optional[str] = None
    "The default GCP project to use when making Vertex API calls."
    location: str = "us-central1"
    "The default location to use when making API calls."
    credentials: Any = None
    "The default custom credentials (google.auth.credentials.Credentials) to use "
    "when making API calls. If not provided, credentials will be ascertained from "
    "the environment."
    request_parallelism: int = 5
    "The amount of parallelism allowed for requests issued to VertexAI models. "
    "Default is 5."
    task_executor: ClassVar[Optional[Executor]] = None

    @property
    def is_codey_model(self) -> bool:
        return is_codey_model(self.model_name)

    @property
    def _default_params(self) -> Dict[str, Any]:
        if self.is_codey_model:
            return {
                "temperature": self.temperature,
                "max_output_tokens": self.max_output_tokens,
            }
        else:
            return {
                "temperature": self.temperature,
                "max_output_tokens": self.max_output_tokens,
                "top_k": self.top_k,
                "top_p": self.top_p,
            }

    def _predict(
        self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any
    ) -> str:
        params = {**self._default_params, **kwargs}
        res = self.client.predict(prompt, **params)
        return self._enforce_stop_words(res.text, stop)

    def _enforce_stop_words(self, text: str, stop: Optional[List[str]] = None) -> str:
        if stop is None and self.stop is not None:
            stop = self.stop
        if stop:
            return enforce_stop_tokens(text, stop)
        return text

    @property
    def _llm_type(self) -> str:
        return "vertexai"

    @classmethod
    def _get_task_executor(cls, request_parallelism: int = 5) -> Executor:
        if cls.task_executor is None:
            cls.task_executor = ThreadPoolExecutor(max_workers=request_parallelism)
        return cls.task_executor

    @classmethod
    def _try_init_vertexai(cls, values: Dict) -> None:
        allowed_params = ["project", "location", "credentials"]
        params = {k: v for k, v in values.items() if k in allowed_params}
        init_vertexai(**params)
        return None


class VertexAI(_VertexAICommon, LLM):
    """Wrapper around Google Vertex AI large language models."""

    model_name: str = "text-bison"
    "The name of the Vertex AI large language model."
    tuned_model_name: Optional[str] = None
    "The name of a tuned model. If provided, model_name is ignored."

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that the python package exists in environment."""
        cls._try_init_vertexai(values)
        tuned_model_name = values.get("tuned_model_name")
        model_name = values["model_name"]
        try:
            if tuned_model_name or not is_codey_model(model_name):
                from vertexai.preview.language_models import TextGenerationModel

                if tuned_model_name:
                    values["client"] = TextGenerationModel.get_tuned_model(
                        tuned_model_name
                    )
                else:
                    values["client"] = TextGenerationModel.from_pretrained(model_name)
            else:
                from vertexai.preview.language_models import CodeGenerationModel

                values["client"] = CodeGenerationModel.from_pretrained(model_name)
        except ImportError:
            raise_vertex_import_error()
        return values

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call Vertex model to get predictions based on the prompt.

        Args:
            prompt: The prompt to pass into the model.
            stop: A list of stop words (optional).
            run_manager: A callback manager for async interaction with LLMs.

        Returns:
            The string generated by the model.
        """
        return await asyncio.wrap_future(
            self._get_task_executor().submit(self._predict, prompt, stop)
        )

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call Vertex model to get predictions based on the prompt.

        Args:
            prompt: The prompt to pass into the model.
            stop: A list of stop words (optional).
            run_manager: A Callbackmanager for LLM run, optional.

        Returns:
            The string generated by the model.
        """
        return self._predict(prompt, stop, **kwargs)
