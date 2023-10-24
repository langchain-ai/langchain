from __future__ import annotations

from concurrent.futures import Executor, ThreadPoolExecutor
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Dict,
    Iterator,
    List,
    Optional,
    Union,
)

from langchain.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain.llms.base import BaseLLM, create_base_retry_decorator
from langchain.pydantic_v1 import BaseModel, Field, root_validator
from langchain.schema import (
    Generation,
    LLMResult,
)
from langchain.schema.output import GenerationChunk
from langchain.utilities.vertexai import (
    init_vertexai,
    raise_vertex_import_error,
)

if TYPE_CHECKING:
    from google.cloud.aiplatform.gapic import (
        PredictionServiceAsyncClient,
        PredictionServiceClient,
    )
    from vertexai.language_models._language_models import (
        TextGenerationResponse,
        _LanguageModel,
    )


def _response_to_generation(
    response: TextGenerationResponse,
) -> GenerationChunk:
    """Convert a stream response to a generation chunk."""
    try:
        generation_info = {
            "is_blocked": response.is_blocked,
            "safety_attributes": response.safety_attributes,
        }
    except Exception:
        generation_info = None
    return GenerationChunk(text=response.text, generation_info=generation_info)


def is_codey_model(model_name: str) -> bool:
    """Returns True if the model name is a Codey model.

    Args:
        model_name: The model name to check.

    Returns: True if the model name is a Codey model.
    """
    return "code" in model_name


def _create_retry_decorator(
    llm: VertexAI,
    *,
    run_manager: Optional[
        Union[AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun]
    ] = None,
) -> Callable[[Any], Any]:
    import google.api_core

    errors = [
        google.api_core.exceptions.ResourceExhausted,
        google.api_core.exceptions.ServiceUnavailable,
        google.api_core.exceptions.Aborted,
        google.api_core.exceptions.DeadlineExceeded,
    ]
    decorator = create_base_retry_decorator(
        error_types=errors, max_retries=llm.max_retries, run_manager=run_manager
    )
    return decorator


def completion_with_retry(
    llm: VertexAI,
    *args: Any,
    run_manager: Optional[CallbackManagerForLLMRun] = None,
    **kwargs: Any,
) -> Any:
    """Use tenacity to retry the completion call."""
    retry_decorator = _create_retry_decorator(llm, run_manager=run_manager)

    @retry_decorator
    def _completion_with_retry(*args: Any, **kwargs: Any) -> Any:
        return llm.client.predict(*args, **kwargs)

    return _completion_with_retry(*args, **kwargs)


def stream_completion_with_retry(
    llm: VertexAI,
    *args: Any,
    run_manager: Optional[CallbackManagerForLLMRun] = None,
    **kwargs: Any,
) -> Any:
    """Use tenacity to retry the completion call."""
    retry_decorator = _create_retry_decorator(llm, run_manager=run_manager)

    @retry_decorator
    def _completion_with_retry(*args: Any, **kwargs: Any) -> Any:
        return llm.client.predict_streaming(*args, **kwargs)

    return _completion_with_retry(*args, **kwargs)


async def acompletion_with_retry(
    llm: VertexAI,
    *args: Any,
    run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
    **kwargs: Any,
) -> Any:
    """Use tenacity to retry the completion call."""
    retry_decorator = _create_retry_decorator(llm, run_manager=run_manager)

    @retry_decorator
    async def _acompletion_with_retry(*args: Any, **kwargs: Any) -> Any:
        return await llm.client.predict_async(*args, **kwargs)

    return await _acompletion_with_retry(*args, **kwargs)


class _VertexAIBase(BaseModel):
    project: Optional[str] = None
    "The default GCP project to use when making Vertex API calls."
    location: str = "us-central1"
    "The default location to use when making API calls."
    request_parallelism: int = 5
    "The amount of parallelism allowed for requests issued to VertexAI models. "
    "Default is 5."
    max_retries: int = 6
    """The maximum number of retries to make when generating."""
    task_executor: ClassVar[Optional[Executor]] = Field(default=None, exclude=True)
    stop: Optional[List[str]] = None
    "Optional list of stop words to use when generating."
    model_name: Optional[str] = None
    "Underlying model name."

    @classmethod
    def _get_task_executor(cls, request_parallelism: int = 5) -> Executor:
        if cls.task_executor is None:
            cls.task_executor = ThreadPoolExecutor(max_workers=request_parallelism)
        return cls.task_executor


class _VertexAICommon(_VertexAIBase):
    client: "_LanguageModel" = None  #: :meta private:
    model_name: str
    "Underlying model name."
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
    credentials: Any = Field(default=None, exclude=True)
    "The default custom credentials (google.auth.credentials.Credentials) to use "
    "when making API calls. If not provided, credentials will be ascertained from "
    "the environment."
    n: int = 1
    """How many completions to generate for each prompt."""
    streaming: bool = False
    """Whether to stream the results or not."""

    @property
    def _llm_type(self) -> str:
        return "vertexai"

    @property
    def is_codey_model(self) -> bool:
        return is_codey_model(self.model_name)

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {**{"model_name": self.model_name}, **self._default_params}

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
                "candidate_count": self.n,
            }

    @classmethod
    def _try_init_vertexai(cls, values: Dict) -> None:
        allowed_params = ["project", "location", "credentials"]
        params = {k: v for k, v in values.items() if k in allowed_params}
        init_vertexai(**params)
        return None

    def _prepare_params(
        self,
        stop: Optional[List[str]] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> dict:
        stop_sequences = stop or self.stop
        params_mapping = {"n": "candidate_count"}
        params = {params_mapping.get(k, k): v for k, v in kwargs.items()}
        params = {**self._default_params, "stop_sequences": stop_sequences, **params}
        if stream or self.streaming:
            params.pop("candidate_count")
        return params


class VertexAI(_VertexAICommon, BaseLLM):
    """Google Vertex AI large language models."""

    model_name: str = "text-bison"
    "The name of the Vertex AI large language model."
    tuned_model_name: Optional[str] = None
    "The name of a tuned model. If provided, model_name is ignored."

    @classmethod
    def is_lc_serializable(self) -> bool:
        return True

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that the python package exists in environment."""
        cls._try_init_vertexai(values)
        tuned_model_name = values.get("tuned_model_name")
        model_name = values["model_name"]
        try:
            if not is_codey_model(model_name):
                from vertexai.preview.language_models import TextGenerationModel

                if tuned_model_name:
                    values["client"] = TextGenerationModel.get_tuned_model(
                        tuned_model_name
                    )
                else:
                    values["client"] = TextGenerationModel.from_pretrained(model_name)
            else:
                from vertexai.preview.language_models import CodeGenerationModel

                if tuned_model_name:
                    values["client"] = CodeGenerationModel.get_tuned_model(
                        tuned_model_name
                    )
                else:
                    values["client"] = CodeGenerationModel.from_pretrained(model_name)
        except ImportError:
            raise_vertex_import_error()

        if values["streaming"] and values["n"] > 1:
            raise ValueError("Only one candidate can be generated with streaming!")
        return values

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> LLMResult:
        should_stream = stream if stream is not None else self.streaming
        params = self._prepare_params(stop=stop, stream=should_stream, **kwargs)
        generations = []
        for prompt in prompts:
            if should_stream:
                generation = GenerationChunk(text="")
                for chunk in self._stream(
                    prompt, stop=stop, run_manager=run_manager, **kwargs
                ):
                    generation += chunk
                generations.append([generation])
            else:
                res = completion_with_retry(
                    self, prompt, run_manager=run_manager, **params
                )
                generations.append([_response_to_generation(r) for r in res.candidates])
        return LLMResult(generations=generations)

    async def _agenerate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        params = self._prepare_params(stop=stop, **kwargs)
        generations = []
        for prompt in prompts:
            res = await acompletion_with_retry(
                self, prompt, run_manager=run_manager, **params
            )
            generations.append([_response_to_generation(r) for r in res.candidates])
        return LLMResult(generations=generations)

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        params = self._prepare_params(stop=stop, stream=True, **kwargs)
        for stream_resp in stream_completion_with_retry(
            self, prompt, run_manager=run_manager, **params
        ):
            chunk = _response_to_generation(stream_resp)
            yield chunk
            if run_manager:
                run_manager.on_llm_new_token(
                    chunk.text,
                    chunk=chunk,
                    verbose=self.verbose,
                )


class VertexAIModelGarden(_VertexAIBase, BaseLLM):
    """Large language models served from Vertex AI Model Garden."""

    client: "PredictionServiceClient" = None  #: :meta private:
    async_client: "PredictionServiceAsyncClient" = None  #: :meta private:
    endpoint_id: str
    "A name of an endpoint where the model has been deployed."
    allowed_model_args: Optional[List[str]] = None
    """Allowed optional args to be passed to the model."""
    prompt_arg: str = "prompt"
    result_arg: str = "generated_text"

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that the python package exists in environment."""
        try:
            from google.api_core.client_options import ClientOptions
            from google.cloud.aiplatform.gapic import (
                PredictionServiceAsyncClient,
                PredictionServiceClient,
            )
        except ImportError:
            raise_vertex_import_error()

        if values["project"] is None:
            raise ValueError(
                "A GCP project should be provided to run inference on Model Garden!"
            )

        client_options = ClientOptions(
            api_endpoint=f"{values['location']}-aiplatform.googleapis.com"
        )
        values["client"] = PredictionServiceClient(client_options=client_options)
        values["async_client"] = PredictionServiceAsyncClient(
            client_options=client_options
        )
        return values

    @property
    def _llm_type(self) -> str:
        return "vertexai_model_garden"

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Run the LLM on the given prompt and input."""
        try:
            from google.protobuf import json_format
            from google.protobuf.struct_pb2 import Value
        except ImportError:
            raise ImportError(
                "protobuf package not found, please install it with"
                " `pip install protobuf`"
            )

        instances = []
        for prompt in prompts:
            if self.allowed_model_args:
                instance = {
                    k: v for k, v in kwargs.items() if k in self.allowed_model_args
                }
            else:
                instance = {}
            instance[self.prompt_arg] = prompt
            instances.append(instance)

        predict_instances = [
            json_format.ParseDict(instance_dict, Value()) for instance_dict in instances
        ]

        endpoint = self.client.endpoint_path(
            project=self.project, location=self.location, endpoint=self.endpoint_id
        )
        response = self.client.predict(endpoint=endpoint, instances=predict_instances)
        generations: List[List[Generation]] = []
        for result in response.predictions:
            generations.append(
                [Generation(text=prediction[self.result_arg]) for prediction in result]
            )
        return LLMResult(generations=generations)

    async def _agenerate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Run the LLM on the given prompt and input."""
        try:
            from google.protobuf import json_format
            from google.protobuf.struct_pb2 import Value
        except ImportError:
            raise ImportError(
                "protobuf package not found, please install it with"
                " `pip install protobuf`"
            )

        instances = []
        for prompt in prompts:
            if self.allowed_model_args:
                instance = {
                    k: v for k, v in kwargs.items() if k in self.allowed_model_args
                }
            else:
                instance = {}
            instance[self.prompt_arg] = prompt
            instances.append(instance)

        predict_instances = [
            json_format.ParseDict(instance_dict, Value()) for instance_dict in instances
        ]

        endpoint = self.async_client.endpoint_path(
            project=self.project, location=self.location, endpoint=self.endpoint_id
        )
        response = await self.async_client.predict(
            endpoint=endpoint, instances=predict_instances
        )
        generations: List[List[Generation]] = []
        for result in response.predictions:
            generations.append(
                [Generation(text=prediction[self.result_arg]) for prediction in result]
            )
        return LLMResult(generations=generations)
