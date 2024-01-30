from __future__ import annotations

from concurrent.futures import Executor, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Iterator, List, Optional, Union

from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import Generation, GenerationChunk, LLMResult
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator

from langchain_community.utilities.vertexai import (
    create_retry_decorator,
    get_client_info,
    init_vertexai,
    raise_vertex_import_error,
)

if TYPE_CHECKING:
    from google.cloud.aiplatform.gapic import (
        PredictionServiceAsyncClient,
        PredictionServiceClient,
    )
    from google.cloud.aiplatform.models import Prediction
    from google.protobuf.struct_pb2 import Value
    from vertexai.language_models._language_models import (
        TextGenerationResponse,
        _LanguageModel,
    )
    from vertexai.preview.generative_models import Image

# This is for backwards compatibility
# We can remove after `langchain` stops importing it
_response_to_generation = None
completion_with_retry = None
stream_completion_with_retry = None


def is_codey_model(model_name: str) -> bool:
    """Returns True if the model name is a Codey model."""
    return "code" in model_name


def is_gemini_model(model_name: str) -> bool:
    """Returns True if the model name is a Gemini model."""
    return model_name is not None and "gemini" in model_name


def completion_with_retry(
    llm: VertexAI,
    prompt: List[Union[str, "Image"]],
    stream: bool = False,
    is_gemini: bool = False,
    run_manager: Optional[CallbackManagerForLLMRun] = None,
    **kwargs: Any,
) -> Any:
    """Use tenacity to retry the completion call."""
    retry_decorator = create_retry_decorator(llm, run_manager=run_manager)

    @retry_decorator
    def _completion_with_retry(
        prompt: List[Union[str, "Image"]], is_gemini: bool = False, **kwargs: Any
    ) -> Any:
        if is_gemini:
            return llm.client.generate_content(
                prompt, stream=stream, generation_config=kwargs
            )
        else:
            if stream:
                return llm.client.predict_streaming(prompt[0], **kwargs)
            return llm.client.predict(prompt[0], **kwargs)

    return _completion_with_retry(prompt, is_gemini, **kwargs)


async def acompletion_with_retry(
    llm: VertexAI,
    prompt: str,
    is_gemini: bool = False,
    run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
    **kwargs: Any,
) -> Any:
    """Use tenacity to retry the completion call."""
    retry_decorator = create_retry_decorator(llm, run_manager=run_manager)

    @retry_decorator
    async def _acompletion_with_retry(
        prompt: str, is_gemini: bool = False, **kwargs: Any
    ) -> Any:
        if is_gemini:
            return await llm.client.generate_content_async(
                prompt, generation_config=kwargs
            )
        return await llm.client.predict_async(prompt, **kwargs)

    return await _acompletion_with_retry(prompt, is_gemini, **kwargs)


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
    client_preview: "_LanguageModel" = None  #: :meta private:
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
    def _is_gemini_model(self) -> bool:
        return is_gemini_model(self.model_name)

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Gets the identifying parameters."""
        return {**{"model_name": self.model_name}, **self._default_params}

    @property
    def _default_params(self) -> Dict[str, Any]:
        params = {
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "candidate_count": self.n,
        }
        if not self.is_codey_model:
            params.update(
                {
                    "top_k": self.top_k,
                    "top_p": self.top_p,
                }
            )
        return params

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


@deprecated(
    since="0.0.12",
    removal="0.2.0",
    alternative_import="langchain_google_vertexai.VertexAI",
)
class VertexAI(_VertexAICommon, BaseLLM):
    """Google Vertex AI large language models."""

    model_name: str = "text-bison"
    "The name of the Vertex AI large language model."
    tuned_model_name: Optional[str] = None
    "The name of a tuned model. If provided, model_name is ignored."

    @classmethod
    def is_lc_serializable(self) -> bool:
        return True

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "llms", "vertexai"]

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that the python package exists in environment."""
        tuned_model_name = values.get("tuned_model_name")
        model_name = values["model_name"]
        is_gemini = is_gemini_model(values["model_name"])
        cls._try_init_vertexai(values)
        try:
            from vertexai.language_models import (
                CodeGenerationModel,
                TextGenerationModel,
            )
            from vertexai.preview.language_models import (
                CodeGenerationModel as PreviewCodeGenerationModel,
            )
            from vertexai.preview.language_models import (
                TextGenerationModel as PreviewTextGenerationModel,
            )

            if is_gemini:
                from vertexai.preview.generative_models import (
                    GenerativeModel,
                )

            if is_codey_model(model_name):
                model_cls = CodeGenerationModel
                preview_model_cls = PreviewCodeGenerationModel
            elif is_gemini:
                model_cls = GenerativeModel
                preview_model_cls = GenerativeModel
            else:
                model_cls = TextGenerationModel
                preview_model_cls = PreviewTextGenerationModel

            if tuned_model_name:
                values["client"] = model_cls.get_tuned_model(tuned_model_name)
                values["client_preview"] = preview_model_cls.get_tuned_model(
                    tuned_model_name
                )
            else:
                if is_gemini:
                    values["client"] = model_cls(model_name=model_name)
                    values["client_preview"] = preview_model_cls(model_name=model_name)
                else:
                    values["client"] = model_cls.from_pretrained(model_name)
                    values["client_preview"] = preview_model_cls.from_pretrained(
                        model_name
                    )

        except ImportError:
            raise_vertex_import_error()

        if values["streaming"] and values["n"] > 1:
            raise ValueError("Only one candidate can be generated with streaming!")
        return values

    def get_num_tokens(self, text: str) -> int:
        """Get the number of tokens present in the text.

        Useful for checking if an input will fit in a model's context window.

        Args:
            text: The string input to tokenize.

        Returns:
            The integer number of tokens in the text.
        """
        try:
            result = self.client_preview.count_tokens([text])
        except AttributeError:
            raise_vertex_import_error()

        return result.total_tokens

    def _response_to_generation(
        self, response: TextGenerationResponse
    ) -> GenerationChunk:
        """Converts a stream response to a generation chunk."""
        try:
            generation_info = {
                "is_blocked": response.is_blocked,
                "safety_attributes": response.safety_attributes,
            }
        except Exception:
            generation_info = None
        return GenerationChunk(text=response.text, generation_info=generation_info)

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
        generations: List[List[Generation]] = []
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
                    self,
                    [prompt],
                    stream=should_stream,
                    is_gemini=self._is_gemini_model,
                    run_manager=run_manager,
                    **params,
                )
                generations.append(
                    [self._response_to_generation(r) for r in res.candidates]
                )
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
                self,
                prompt,
                is_gemini=self._is_gemini_model,
                run_manager=run_manager,
                **params,
            )
            generations.append(
                [self._response_to_generation(r) for r in res.candidates]
            )
        return LLMResult(generations=generations)

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        params = self._prepare_params(stop=stop, stream=True, **kwargs)
        for stream_resp in completion_with_retry(
            self,
            [prompt],
            stream=True,
            is_gemini=self._is_gemini_model,
            run_manager=run_manager,
            **params,
        ):
            chunk = self._response_to_generation(stream_resp)
            yield chunk
            if run_manager:
                run_manager.on_llm_new_token(
                    chunk.text,
                    chunk=chunk,
                    verbose=self.verbose,
                )


@deprecated(
    since="0.0.12",
    removal="0.2.0",
    alternative_import="langchain_google_vertexai.VertexAIModelGarden",
)
class VertexAIModelGarden(_VertexAIBase, BaseLLM):
    """Large language models served from Vertex AI Model Garden."""

    client: "PredictionServiceClient" = None  #: :meta private:
    async_client: "PredictionServiceAsyncClient" = None  #: :meta private:
    endpoint_id: str
    "A name of an endpoint where the model has been deployed."
    allowed_model_args: Optional[List[str]] = None
    "Allowed optional args to be passed to the model."
    prompt_arg: str = "prompt"
    result_arg: Optional[str] = "generated_text"
    "Set result_arg to None if output of the model is expected to be a string."
    "Otherwise, if it's a dict, provided an argument that contains the result."

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

        if not values["project"]:
            raise ValueError(
                "A GCP project should be provided to run inference on Model Garden!"
            )

        client_options = ClientOptions(
            api_endpoint=f"{values['location']}-aiplatform.googleapis.com"
        )
        client_info = get_client_info(module="vertex-ai-model-garden")
        values["client"] = PredictionServiceClient(
            client_options=client_options, client_info=client_info
        )
        values["async_client"] = PredictionServiceAsyncClient(
            client_options=client_options, client_info=client_info
        )
        return values

    @property
    def endpoint_path(self) -> str:
        return self.client.endpoint_path(
            project=self.project,
            location=self.location,
            endpoint=self.endpoint_id,
        )

    @property
    def _llm_type(self) -> str:
        return "vertexai_model_garden"

    def _prepare_request(self, prompts: List[str], **kwargs: Any) -> List["Value"]:
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
        return predict_instances

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Run the LLM on the given prompt and input."""
        instances = self._prepare_request(prompts, **kwargs)
        response = self.client.predict(endpoint=self.endpoint_path, instances=instances)
        return self._parse_response(response)

    def _parse_response(self, predictions: "Prediction") -> LLMResult:
        generations: List[List[Generation]] = []
        for result in predictions.predictions:
            generations.append(
                [
                    Generation(text=self._parse_prediction(prediction))
                    for prediction in result
                ]
            )
        return LLMResult(generations=generations)

    def _parse_prediction(self, prediction: Any) -> str:
        if isinstance(prediction, str):
            return prediction

        if self.result_arg:
            try:
                return prediction[self.result_arg]
            except KeyError:
                if isinstance(prediction, str):
                    error_desc = (
                        "Provided non-None `result_arg` (result_arg="
                        f"{self.result_arg}). But got prediction of type "
                        f"{type(prediction)} instead of dict. Most probably, you"
                        "need to set `result_arg=None` during VertexAIModelGarden "
                        "initialization."
                    )
                    raise ValueError(error_desc)
                else:
                    raise ValueError(f"{self.result_arg} key not found in prediction!")

        return prediction

    async def _agenerate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Run the LLM on the given prompt and input."""
        instances = self._prepare_request(prompts, **kwargs)
        response = await self.async_client.predict(
            endpoint=self.endpoint_path, instances=instances
        )
        return self._parse_response(response)
