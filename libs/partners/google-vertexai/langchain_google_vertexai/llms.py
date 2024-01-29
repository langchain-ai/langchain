from __future__ import annotations

from concurrent.futures import Executor
from typing import Any, ClassVar, Dict, Iterator, List, Optional, Union

import vertexai  # type: ignore[import-untyped]
from google.api_core.client_options import ClientOptions
from google.cloud.aiplatform.gapic import (
    PredictionServiceAsyncClient,
    PredictionServiceClient,
)
from google.cloud.aiplatform.models import Prediction
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import Generation, GenerationChunk, LLMResult
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator
from vertexai.language_models import (  # type: ignore[import-untyped]
    CodeGenerationModel,
    TextGenerationModel,
)
from vertexai.language_models._language_models import (  # type: ignore[import-untyped]
    TextGenerationResponse,
)
from vertexai.preview.generative_models import (  # type: ignore[import-untyped]
    GenerativeModel,
    Image,
)
from vertexai.preview.language_models import (  # type: ignore[import-untyped]
    CodeGenerationModel as PreviewCodeGenerationModel,
)
from vertexai.preview.language_models import (
    TextGenerationModel as PreviewTextGenerationModel,
)

from langchain_google_vertexai._enums import HarmBlockThreshold, HarmCategory
from langchain_google_vertexai._utils import (
    create_retry_decorator,
    get_client_info,
    get_generation_info,
    is_codey_model,
    is_gemini_model,
)

_PALM_DEFAULT_MAX_OUTPUT_TOKENS = TextGenerationModel._DEFAULT_MAX_OUTPUT_TOKENS
_PALM_DEFAULT_TEMPERATURE = 0.0
_PALM_DEFAULT_TOP_P = 0.95
_PALM_DEFAULT_TOP_K = 40


def _completion_with_retry(
    llm: VertexAI,
    prompt: List[Union[str, Image]],
    stream: bool = False,
    is_gemini: bool = False,
    run_manager: Optional[CallbackManagerForLLMRun] = None,
    **kwargs: Any,
) -> Any:
    """Use tenacity to retry the completion call."""
    retry_decorator = create_retry_decorator(
        max_retries=llm.max_retries, run_manager=run_manager
    )

    @retry_decorator
    def _completion_with_retry_inner(
        prompt: List[Union[str, Image]], is_gemini: bool = False, **kwargs: Any
    ) -> Any:
        if is_gemini:
            return llm.client.generate_content(
                prompt,
                stream=stream,
                safety_settings=kwargs.pop("safety_settings", None),
                generation_config=kwargs,
            )
        else:
            if stream:
                return llm.client.predict_streaming(prompt[0], **kwargs)
            return llm.client.predict(prompt[0], **kwargs)

    return _completion_with_retry_inner(prompt, is_gemini, **kwargs)


async def _acompletion_with_retry(
    llm: VertexAI,
    prompt: str,
    is_gemini: bool = False,
    run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
    **kwargs: Any,
) -> Any:
    """Use tenacity to retry the completion call."""
    retry_decorator = create_retry_decorator(
        max_retries=llm.max_retries, run_manager=run_manager
    )

    @retry_decorator
    async def _acompletion_with_retry_inner(
        prompt: str, is_gemini: bool = False, **kwargs: Any
    ) -> Any:
        if is_gemini:
            return await llm.client.generate_content_async(
                prompt,
                generation_config=kwargs,
                safety_settings=kwargs.pop("safety_settings", None),
            )
        return await llm.client.predict_async(prompt, **kwargs)

    return await _acompletion_with_retry_inner(prompt, is_gemini, **kwargs)


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


class _VertexAICommon(_VertexAIBase):
    client: Any = None  #: :meta private:
    client_preview: Any = None  #: :meta private:
    model_name: str
    "Underlying model name."
    temperature: Optional[float] = None
    "Sampling temperature, it controls the degree of randomness in token selection."
    max_output_tokens: Optional[int] = None
    "Token limit determines the maximum amount of text output from one prompt."
    top_p: Optional[float] = None
    "Tokens are selected from most probable to least until the sum of their "
    "probabilities equals the top-p value. Top-p is ignored for Codey models."
    top_k: Optional[int] = None
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
    safety_settings: Optional[Dict[HarmCategory, HarmBlockThreshold]] = None
    """The default safety settings to use for all generations. 
    
        For example: 

            from langchain_google_vertexai import HarmBlockThreshold, HarmCategory

            safety_settings = {
                HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            }
            """  # noqa: E501

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
        if self._is_gemini_model:
            default_params = {}
        else:
            default_params = {
                "temperature": _PALM_DEFAULT_TEMPERATURE,
                "max_output_tokens": _PALM_DEFAULT_MAX_OUTPUT_TOKENS,
                "top_p": _PALM_DEFAULT_TOP_P,
                "top_k": _PALM_DEFAULT_TOP_K,
            }
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
        updated_params = {}
        for param_name, param_value in params.items():
            default_value = default_params.get(param_name)
            if param_value or default_value:
                updated_params[param_name] = (
                    param_value if param_value else default_value
                )
        return updated_params

    @classmethod
    def _init_vertexai(cls, values: Dict) -> None:
        vertexai.init(
            project=values.get("project"),
            location=values.get("location"),
            credentials=values.get("credentials"),
        )
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

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "llms", "vertexai"]

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that the python package exists in environment."""
        tuned_model_name = values.get("tuned_model_name")
        model_name = values["model_name"]
        safety_settings = values["safety_settings"]
        is_gemini = is_gemini_model(values["model_name"])
        cls._init_vertexai(values)

        if safety_settings and (not is_gemini or tuned_model_name):
            raise ValueError("Safety settings are only supported for Gemini models")

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
                values["client"] = model_cls(
                    model_name=model_name, safety_settings=safety_settings
                )
                values["client_preview"] = preview_model_cls(
                    model_name=model_name, safety_settings=safety_settings
                )
            else:
                values["client"] = model_cls.from_pretrained(model_name)
                values["client_preview"] = preview_model_cls.from_pretrained(model_name)

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
        result = self.client_preview.count_tokens([text])
        return result.total_tokens

    def _response_to_generation(
        self, response: TextGenerationResponse, *, stream: bool = False
    ) -> GenerationChunk:
        """Converts a stream response to a generation chunk."""
        generation_info = get_generation_info(
            response, self._is_gemini_model, stream=stream
        )
        try:
            text = response.text
        except AttributeError:
            text = ""
        except ValueError:
            text = ""
        return GenerationChunk(
            text=text,
            generation_info=generation_info,
        )

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
                res = _completion_with_retry(
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
        generations: List[List[Generation]] = []
        for prompt in prompts:
            res = await _acompletion_with_retry(
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
        for stream_resp in _completion_with_retry(
            self,
            [prompt],
            stream=True,
            is_gemini=self._is_gemini_model,
            run_manager=run_manager,
            **params,
        ):
            # Gemini models return GenerationResponse even when streaming, which has a
            # candidates field.
            stream_resp = (
                stream_resp
                if isinstance(stream_resp, TextGenerationResponse)
                else stream_resp.candidates[0]
            )
            chunk = self._response_to_generation(stream_resp, stream=True)
            yield chunk
            if run_manager:
                run_manager.on_llm_new_token(
                    chunk.text,
                    chunk=chunk,
                    verbose=self.verbose,
                )


class VertexAIModelGarden(_VertexAIBase, BaseLLM):
    """Large language models served from Vertex AI Model Garden."""

    client: Any = None  #: :meta private:
    async_client: Any = None  #: :meta private:
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
            project=self.project, location=self.location, endpoint=self.endpoint_id
        )

    @property
    def _llm_type(self) -> str:
        return "vertexai_model_garden"

    def _prepare_request(self, prompts: List[str], **kwargs: Any) -> List["Value"]:
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
