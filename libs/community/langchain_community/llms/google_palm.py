from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional

from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LanguageModelInput
from langchain_core.outputs import Generation, GenerationChunk, LLMResult
from langchain_core.utils import get_from_dict_or_env, pre_init
from pydantic import BaseModel, SecretStr

from langchain_community.llms import BaseLLM
from langchain_community.utilities.vertexai import create_retry_decorator


def completion_with_retry(
    llm: GooglePalm,
    prompt: LanguageModelInput,
    is_gemini: bool = False,
    stream: bool = False,
    run_manager: Optional[CallbackManagerForLLMRun] = None,
    **kwargs: Any,
) -> Any:
    """Use tenacity to retry the completion call."""
    retry_decorator = create_retry_decorator(
        llm, max_retries=llm.max_retries, run_manager=run_manager
    )

    @retry_decorator
    def _completion_with_retry(
        prompt: LanguageModelInput, is_gemini: bool, stream: bool, **kwargs: Any
    ) -> Any:
        generation_config = kwargs.get("generation_config", {})
        if is_gemini:
            return llm.client.generate_content(
                contents=prompt, stream=stream, generation_config=generation_config
            )
        return llm.client.generate_text(prompt=prompt, **kwargs)

    return _completion_with_retry(
        prompt=prompt, is_gemini=is_gemini, stream=stream, **kwargs
    )


def _is_gemini_model(model_name: str) -> bool:
    return "gemini" in model_name


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


@deprecated("0.0.12", alternative_import="langchain_google_genai.GoogleGenerativeAI")
class GooglePalm(BaseLLM, BaseModel):
    """
    DEPRECATED: Use `langchain_google_genai.GoogleGenerativeAI` instead.

    Google PaLM models.
    """

    client: Any  #: :meta private:
    google_api_key: Optional[SecretStr]
    model_name: str = "models/text-bison-001"
    """Model name to use."""
    temperature: float = 0.7
    """Run inference with this temperature. Must be in the closed interval
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

    @property
    def is_gemini(self) -> bool:
        """Returns whether a model is belongs to a Gemini family or not."""
        return _is_gemini_model(self.model_name)

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"google_api_key": "GOOGLE_API_KEY"}

    @classmethod
    def is_lc_serializable(self) -> bool:
        return True

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "llms", "google_palm"]

    @pre_init
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate api key, python package exists."""
        google_api_key = get_from_dict_or_env(
            values, "google_api_key", "GOOGLE_API_KEY"
        )
        model_name = values["model_name"]
        try:
            import google.generativeai as genai

            if isinstance(google_api_key, SecretStr):
                google_api_key = google_api_key.get_secret_value()

            genai.configure(api_key=google_api_key)

            if _is_gemini_model(model_name):
                values["client"] = genai.GenerativeModel(model_name=model_name)
            else:
                values["client"] = genai
        except ImportError:
            raise ImportError(
                "Could not import google-generativeai python package. "
                "Please install it with `pip install google-generativeai`."
            )

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
            if self.is_gemini:
                res = completion_with_retry(
                    self,
                    prompt=prompt,
                    stream=False,
                    is_gemini=True,
                    run_manager=run_manager,
                    generation_config=generation_config,
                )
                candidates = [
                    "".join([p.text for p in c.content.parts]) for c in res.candidates
                ]
                generations.append([Generation(text=c) for c in candidates])
            else:
                res = completion_with_retry(
                    self,
                    model=self.model_name,
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
        for stream_resp in completion_with_retry(
            self,
            prompt,
            stream=True,
            is_gemini=True,
            run_manager=run_manager,
            generation_config=generation_config,
            **kwargs,
        ):
            chunk = GenerationChunk(text=stream_resp.text)
            if run_manager:
                run_manager.on_llm_new_token(
                    stream_resp.text,
                    chunk=chunk,
                    verbose=self.verbose,
                )
            yield chunk

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
        if self.is_gemini:
            raise ValueError("Counting tokens is not yet supported!")
        result = self.client.count_text_tokens(model=self.model_name, prompt=text)
        return result["token_count"]
