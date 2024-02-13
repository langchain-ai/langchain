from __future__ import annotations

from enum import Enum, auto
from typing import Any, Callable, Dict, Iterator, List, Optional, Union

import google.api_core
import google.generativeai as genai  # type: ignore[import]
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.llms import BaseLLM, create_base_retry_decorator
from langchain_core.outputs import Generation, GenerationChunk, LLMResult
from langchain_core.pydantic_v1 import BaseModel, Field, SecretStr, root_validator
from langchain_core.utils import get_from_dict_or_env

from langchain_google_genai._enums import (
    HarmBlockThreshold,
    HarmCategory,
)


class GoogleModelFamily(str, Enum):
    GEMINI = auto()
    PALM = auto()

    @classmethod
    def _missing_(cls, value: Any) -> Optional["GoogleModelFamily"]:
        if "gemini" in value.lower():
            return GoogleModelFamily.GEMINI
        elif "text-bison" in value.lower():
            return GoogleModelFamily.PALM
        return None


def _create_retry_decorator(
    llm: BaseLLM,
    *,
    max_retries: int = 1,
    run_manager: Optional[
        Union[AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun]
    ] = None,
) -> Callable[[Any], Any]:
    """Creates a retry decorator for Vertex / Palm LLMs."""

    errors = [
        google.api_core.exceptions.ResourceExhausted,
        google.api_core.exceptions.ServiceUnavailable,
        google.api_core.exceptions.Aborted,
        google.api_core.exceptions.DeadlineExceeded,
        google.api_core.exceptions.GoogleAPIError,
    ]
    decorator = create_base_retry_decorator(
        error_types=errors, max_retries=max_retries, run_manager=run_manager
    )
    return decorator


def _completion_with_retry(
    llm: GoogleGenerativeAI,
    prompt: LanguageModelInput,
    is_gemini: bool = False,
    stream: bool = False,
    run_manager: Optional[CallbackManagerForLLMRun] = None,
    **kwargs: Any,
) -> Any:
    """Use tenacity to retry the completion call."""
    retry_decorator = _create_retry_decorator(
        llm, max_retries=llm.max_retries, run_manager=run_manager
    )

    @retry_decorator
    def _completion_with_retry(
        prompt: LanguageModelInput, is_gemini: bool, stream: bool, **kwargs: Any
    ) -> Any:
        generation_config = kwargs.get("generation_config", {})
        error_msg = (
            "Your location is not supported by google-generativeai at the moment. "
            "Try to use VertexAI LLM from langchain_google_vertexai"
        )
        try:
            if is_gemini:
                return llm.client.generate_content(
                    contents=prompt,
                    stream=stream,
                    generation_config=generation_config,
                    safety_settings=kwargs.pop("safety_settings", None),
                )
            return llm.client.generate_text(prompt=prompt, **kwargs)
        except google.api_core.exceptions.FailedPrecondition as exc:
            if "location is not supported" in exc.message:
                raise ValueError(error_msg)

    return _completion_with_retry(
        prompt=prompt, is_gemini=is_gemini, stream=stream, **kwargs
    )


def _strip_erroneous_leading_spaces(text: str) -> str:
    """Strip erroneous leading spaces from text.

    The PaLM API will sometimes erroneously return a single leading space in all
    lines > 1. This function strips that space.
    """
    has_leading_space = all(not line or line[0] == " " for line in text.split("\n")[1:])
    if has_leading_space:
        return text.replace("\n ", "\n")
    else:
        return text


class _BaseGoogleGenerativeAI(BaseModel):
    """Base class for Google Generative AI LLMs"""

    model: str = Field(
        ...,
        description="""The name of the model to use.
Supported examples:
    - gemini-pro
    - models/text-bison-001""",
    )
    """Model name to use."""
    google_api_key: Optional[SecretStr] = None
    temperature: float = 0.7
    """Run inference with this temperature. Must by in the closed interval
       [0.0, 1.0]."""
    top_p: Optional[float] = None
    """Decode using nucleus sampling: consider the smallest set of tokens whose
       probability sum is at least top_p. Must be in the closed interval [0.0, 1.0]."""
    top_k: Optional[int] = None
    """Decode using top-k sampling: consider the set of top_k most probable tokens.
       Must be positive."""
    max_output_tokens: Optional[int] = None
    """Maximum number of tokens to include in a candidate. Must be greater than zero.
       If unset, will default to 64."""
    n: int = 1
    """Number of chat completions to generate for each prompt. Note that the API may
       not return the full n completions if duplicates are generated."""
    max_retries: int = 6
    """The maximum number of retries to make when generating."""
    client_options: Optional[Dict] = Field(
        None,
        description=(
            "A dictionary of client options to pass to the Google API client, "
            "such as `api_endpoint`."
        ),
    )
    transport: Optional[str] = Field(
        None,
        description="A string, one of: [`rest`, `grpc`, `grpc_asyncio`].",
    )

    safety_settings: Optional[Dict[HarmCategory, HarmBlockThreshold]] = None
    """The default safety settings to use for all generations. 
    
        For example: 

            from google.generativeai.types.safety_types import HarmBlockThreshold, HarmCategory

            safety_settings = {
                HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            }
            """  # noqa: E501

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"google_api_key": "GOOGLE_API_KEY"}

    @property
    def _model_family(self) -> str:
        return GoogleModelFamily(self.model)

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_output_tokens": self.max_output_tokens,
            "candidate_count": self.n,
        }


class GoogleGenerativeAI(_BaseGoogleGenerativeAI, BaseLLM):
    """Google GenerativeAI models.

    Example:
        .. code-block:: python

            from langchain_google_genai import GoogleGenerativeAI
            llm = GoogleGenerativeAI(model="gemini-pro")
    """

    client: Any  #: :meta private:

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validates params and passes them to google-generativeai package."""
        google_api_key = get_from_dict_or_env(
            values, "google_api_key", "GOOGLE_API_KEY"
        )
        model_name = values["model"]

        safety_settings = values["safety_settings"]

        if isinstance(google_api_key, SecretStr):
            google_api_key = google_api_key.get_secret_value()

        genai.configure(
            api_key=google_api_key,
            transport=values.get("transport"),
            client_options=values.get("client_options"),
        )

        if safety_settings and (
            not GoogleModelFamily(model_name) == GoogleModelFamily.GEMINI
        ):
            raise ValueError("Safety settings are only supported for Gemini models")

        if GoogleModelFamily(model_name) == GoogleModelFamily.GEMINI:
            values["client"] = genai.GenerativeModel(
                model_name=model_name, safety_settings=safety_settings
            )
        else:
            values["client"] = genai

        if values["temperature"] is not None and not 0 <= values["temperature"] <= 1:
            raise ValueError("temperature must be in the range [0.0, 1.0]")

        if values["top_p"] is not None and not 0 <= values["top_p"] <= 1:
            raise ValueError("top_p must be in the range [0.0, 1.0]")

        if values["top_k"] is not None and values["top_k"] <= 0:
            raise ValueError("top_k must be positive")

        if values["max_output_tokens"] is not None and values["max_output_tokens"] <= 0:
            raise ValueError("max_output_tokens must be greater than zero")

        return values

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        generations: List[List[Generation]] = []
        generation_config = {
            "stop_sequences": stop,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_output_tokens": self.max_output_tokens,
            "candidate_count": self.n,
        }
        for prompt in prompts:
            if self._model_family == GoogleModelFamily.GEMINI:
                res = _completion_with_retry(
                    self,
                    prompt=prompt,
                    stream=False,
                    is_gemini=True,
                    run_manager=run_manager,
                    generation_config=generation_config,
                    safety_settings=kwargs.pop("safety_settings", None),
                )
                candidates = [
                    "".join([p.text for p in c.content.parts]) for c in res.candidates
                ]
                generations.append([Generation(text=c) for c in candidates])
            else:
                res = _completion_with_retry(
                    self,
                    model=self.model,
                    prompt=prompt,
                    stream=False,
                    is_gemini=False,
                    run_manager=run_manager,
                    **generation_config,
                )
                prompt_generations = []
                for candidate in res.candidates:
                    raw_text = candidate["output"]
                    stripped_text = _strip_erroneous_leading_spaces(raw_text)
                    prompt_generations.append(Generation(text=stripped_text))
                generations.append(prompt_generations)

        return LLMResult(generations=generations)

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        generation_config = kwargs.get("generation_config", {})
        if stop:
            generation_config["stop_sequences"] = stop
        for stream_resp in _completion_with_retry(
            self,
            prompt,
            stream=True,
            is_gemini=True,
            run_manager=run_manager,
            generation_config=generation_config,
            safety_settings=kwargs.pop("safety_settings", None),
            **kwargs,
        ):
            chunk = GenerationChunk(text=stream_resp.text)
            yield chunk
            if run_manager:
                run_manager.on_llm_new_token(
                    stream_resp.text,
                    chunk=chunk,
                    verbose=self.verbose,
                )

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "google_palm"

    def get_num_tokens(self, text: str) -> int:
        """Get the number of tokens present in the text.

        Useful for checking if an input will fit in a model's context window.

        Args:
            text: The string input to tokenize.

        Returns:
            The integer number of tokens in the text.
        """
        if self._model_family == GoogleModelFamily.GEMINI:
            result = self.client.count_tokens(text)
            token_count = result.total_tokens
        else:
            result = self.client.count_text_tokens(model=self.model, prompt=text)
            token_count = result["token_count"]

        return token_count
