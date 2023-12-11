from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import BaseMessage
from langchain_core.outputs import Generation, GenerationChunk, LLMResult
from langchain_core.prompt_values import StringPromptValue
from langchain_core.pydantic_v1 import BaseModel, root_validator
from langchain_core.utils import get_from_dict_or_env

from langchain_community.llms import BaseLLM
from langchain_community.utilities.vertexai import create_retry_decorator

if TYPE_CHECKING:
    from google.ai.generativelanguage import Blob
    from langchain_core.runnables import RunnableConfig


def _image_to_blob(img_path: str, mime_type: Optional[str] = None) -> "Blob":
    from google.ai.generativelanguage import Blob

    if img_path.startswith("gs://"):
        from google.cloud import storage

        gcs_client = storage.Client()
        pieces = img_path.split("/")
        blobs = list(gcs_client.list_blobs(pieces[2], prefix="/".join(pieces[3:])))
        if len(blobs) > 1:
            raise ValueError(f"Found more than one candidate for {img_path}!")
        raw_bytes = blobs[0].download_as_bytes()
    else:
        with open(img_path, "rb") as input_file:
            raw_bytes = input_file.read()
    return Blob(data=raw_bytes, mime_type=mime_type)


def completion_with_retry(
    llm: GoogleGenerativeAI,
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


class GoogleGenerativeAI(BaseLLM, BaseModel):
    """Google GenerativeAI models."""

    client: Any  #: :meta private:
    google_api_key: Optional[str]
    model_name: str = "models/text-bison-001"
    """Model name to use."""
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
    streaming: bool = False
    """Whether to stream the results or not."""

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

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate api key, python package exists."""
        google_api_key = get_from_dict_or_env(
            values, "google_api_key", "GOOGLE_API_KEY"
        )
        model_name = values["model_name"]
        try:
            import google.generativeai as genai

            genai.configure(api_key=google_api_key)

            if _is_gemini_model(model_name):
                values["client"] = genai.GenerativeModel(model_name=model_name)
            else:
                values["client"] = genai
                if values["streaming"]:
                    raise ValueError(
                        "Streaming is not supported for Palm2 models. Use "
                        "langchain.llms.VertexAI if you need streaming!"
                    )
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
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> LLMResult:
        should_stream = stream if stream is not None else self.streaming

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
            if should_stream:
                if not self.is_gemini:
                    raise ValueError(
                        "Streaming is not supported for Palm2 models. Use "
                        "langchain.llms.VertexAI if you need streaming!"
                    )
                generation = GenerationChunk(text="")
                for chunk in self._stream(
                    prompt,
                    stream=True,
                    stop=stop,
                    run_manager=run_manager,
                    generation_config=generation_config,
                    **kwargs,
                ):
                    generation += chunk
                generations.append([generation])
            else:
                if self.is_gemini:
                    res = completion_with_retry(
                        self,
                        prompt=prompt,
                        stream=should_stream,
                        is_gemini=True,
                        run_manager=run_manager,
                        generation_config=generation_config,
                    )
                    candidates = [
                        "".join([p.text for p in c.content.parts])
                        for c in res.candidates
                    ]
                    generations.append([Generation(text=c) for c in candidates])
                else:
                    res = completion_with_retry(
                        self,
                        model=self.model_name,
                        prompt=prompt,
                        stream=should_stream,
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
            yield chunk
            if run_manager:
                run_manager.on_llm_new_token(
                    stream_resp.text,
                    chunk=chunk,
                    verbose=self.verbose,
                )

    def invoke(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        generation_config = {
            "stop_sequences": stop,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_output_tokens": self.max_output_tokens,
            "candidate_count": self.n,
        }
        if isinstance(input, str):
            return super().invoke(input, config, stop=stop, **generation_config)
        elif isinstance(input, StringPromptValue):
            return super().invoke(
                input.to_string(), config, stop=stop, **generation_config
            )
        elif isinstance(input, list):
            if "vision" not in self.model_name:
                raise ValueError(
                    f"Multi-modal input is not supported by {self.model_name} model!"
                )
            first_message = input[0]
            if len(input) > 1 or not isinstance(first_message, BaseMessage):
                raise ValueError(
                    "Multi-modal model expects only a single message as a input!"
                )
        else:
            raise ValueError(
                f"Invalid input type {type(input)}. "
                "Must be a PromptValue, str, or list of BaseMessages."
            )

        messages = []
        for content in first_message.content:
            if not isinstance(content, Dict):
                raise ValueError(
                    f"Message's content is expected to be a dict, got {type(content)}!"
                )
            if content["type"] == "text":
                messages.append(content["text"])
            elif content["type"] == "image_url":
                path = content["image_url"]["url"]
                mime_type = content["image_url"].get("mime_type")
                blob = _image_to_blob(img_path=path, mime_type=mime_type)
                messages.append(blob)
            else:
                raise ValueError("Only text and image_url types are supported!")

        res = completion_with_retry(
            self,
            prompt=messages,
            is_gemini=True,
            stream=False,
            run_manager=kwargs.get("run_manager"),
            **generation_config,
        )

        return res.text

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


class GooglePalm(GoogleGenerativeAI):
    """`GooglePalm` retriever alias for backwards compatibility.
    DEPRECATED: Use `GoogleGenerativeAI` instead.
    """
