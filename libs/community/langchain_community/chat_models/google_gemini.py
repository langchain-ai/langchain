from __future__ import annotations

import asyncio
import base64
import logging
import os
from io import BytesIO
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Type,
    Union,
)
from urllib.parse import urlparse

import requests
from langchain.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain.pydantic_v1 import Field, root_validator
from langchain.utils import get_from_dict_or_env
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ChatMessage,
    ChatMessageChunk,
    HumanMessage,
    HumanMessageChunk,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import google.generativeai as genai

try:
    import PIL.Image
    from PIL import Image
except ImportError:
    PIL = None
    Image = None

try:
    import IPython.display
except ImportError:
    IPython = None


class ChatGoogleGeminiError(Exception):
    """
    Custom exception class for errors associated with the `Google Gemini` API.

    This exception is raised when there are specific issues related to the
    Google Gemini API usage in the ChatGoogleGemini class, such as unsupported
    message types or roles.
    """

    pass


def _create_retry_decorator() -> Callable[[Any], Any]:
    """
    Creates and returns a preconfigured tenacity retry decorator.

    The retry decorator is configured to handle specific Google API exceptions
    such as ResourceExhausted and ServiceUnavailable. It uses an exponential
    backoff strategy for retries.

    Returns:
        Callable[[Any], Any]: A retry decorator configured for handling specific
        Google API exceptions.
    """
    import google.api_core.exceptions

    multiplier = 2
    min_seconds = 1
    max_seconds = 60
    max_retries = 10

    return retry(
        reraise=True,
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=multiplier, min=min_seconds, max=max_seconds),
        retry=(
            retry_if_exception_type(google.api_core.exceptions.ResourceExhausted)
            | retry_if_exception_type(google.api_core.exceptions.ServiceUnavailable)
            | retry_if_exception_type(google.api_core.exceptions.GoogleAPIError)
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )


def chat_with_retry(*, generation_method: Callable, **kwargs: Any) -> Any:
    """
    Executes a chat generation method with retry logic using tenacity.

    This function is a wrapper that applies a retry mechanism to a provided
    chat generation function. It is useful for handling intermittent issues
    like network errors or temporary service unavailability.

    Args:
        generation_method (Callable): The chat generation method to be executed.
        **kwargs (Any): Additional keyword arguments to pass to the generation method.

    Returns:
        Any: The result from the chat generation method.
    """
    retry_decorator = _create_retry_decorator()

    @retry_decorator
    def _chat_with_retry(**kwargs: Any) -> Any:
        return generation_method(**kwargs)

    return _chat_with_retry(**kwargs)


async def achat_with_retry(*, generation_method: Awaitable, **kwargs: Any) -> Any:
    """
    Asynchronously executes a chat generation method with retry logic.

    Similar to `chat_with_retry`, this function applies a retry decorator for
    asynchronous chat generation methods. It handles retries for tasks like
    generating responses from a language model.

    Args:
        generation_method (Awaitable): The async chat generation method to be executed.
        **kwargs (Any): Additional keyword arguments to pass to the generation method.

    Returns:
        Any: The result from the async chat generation method.
    """
    retry_decorator = _create_retry_decorator()

    @retry_decorator
    async def _achat_with_retry(**kwargs: Any) -> Any:
        return await generation_method(**kwargs)

    return await _achat_with_retry(**kwargs)


def _get_role(message: BaseMessage) -> str:
    if isinstance(message, ChatMessage):
        if message.role not in ("user", "model"):
            raise ChatGoogleGeminiError(
                "Gemini only supports user and model roles when"
                " providing it with Chat messages."
            )
        return message.role
    elif isinstance(message, HumanMessage):
        return "user"
    elif isinstance(message, AIMessage):
        return "model"
    else:
        # TODO: Gemini doesn't seem to have a concept of system messages yet.
        raise ChatGoogleGeminiError(
            f"Message of '{message.type}' type not supported by Gemini."
            " Please only provide it with Human or AI (user/assistant) messages."
        )


def _is_openai_parts_format(part: dict) -> bool:
    return "type" in part


def _is_url(s: str) -> bool:
    try:
        result = urlparse(s)
        return all([result.scheme, result.netloc])
    except Exception as e:
        logger.debug(f"Unable to parse URL: {e}")
        return False


def _is_b64(s: str) -> bool:
    return s.startswith("data:image")


def _url_to_pil(image_source: str) -> Image:
    if PIL is None:
        raise ImportError(
            "PIL is required to load images. Please install it "
            "with `pip install pillow`"
        )
    try:
        if isinstance(image_source, (Image.Image, IPython.display.Image)):
            return image_source
        elif _is_url(image_source):
            response = requests.get(image_source)
            response.raise_for_status()
            return Image.open(BytesIO(response.content))
        elif _is_b64(image_source):
            _, encoded = image_source.split(",", 1)
            data = base64.b64decode(encoded)
            return Image.open(BytesIO(data))
        elif os.path.exists(image_source):
            return Image.open(image_source)
        else:
            raise ValueError(
                "The provided string is not a valid URL, base64, or file path."
            )
    except Exception as e:
        raise ValueError(f"Unable to process the provided image source: {e}")


def _convert_to_parts(
    content: Sequence[Union[str, dict]]
) -> List[genai.types.PartType]:
    """Converts a list of LangChain messages into a google parts."""
    import google.generativeai as genai

    parts = []
    for part in content:
        if isinstance(part, str):
            parts.append(genai.types.PartDict(text=part, inline_data=None))
        elif isinstance(part, Mapping):
            # OpenAI Format
            if _is_openai_parts_format(part):
                if part["type"] == "text":
                    parts.append({"text": part["text"]})
                elif part["type"] == "image_url":
                    img_url = part["image_url"]
                    if isinstance(img_url, dict):
                        if "url" not in img_url:
                            raise ValueError(
                                f"Unrecognized message image format: {img_url}"
                            )
                        img_url = img_url["url"]

                    parts.append({"inline_data": _url_to_pil(img_url)})
                else:
                    raise ValueError(f"Unrecognized message part type: {part['type']}")
            else:
                # Yolo
                logger.warning(
                    "Unrecognized message part format. Assuming it's a text part."
                )
                parts.append(part)
        else:
            # TODO: Maybe some of Google's native stuff
            # would hit this branch.
            raise ChatGoogleGeminiError(
                "Gemini only supports text and inline_data parts."
            )
    return parts


def _messages_to_genai_contents(
    input_messages: List[BaseMessage],
) -> List[genai.types.ContentDict]:
    """Converts a list of messages into a Gemini API google content dicts."""

    messages: List[genai.types.MessageDict] = []

    for i, message in enumerate(input_messages):
        role = _get_role(message)
        if isinstance(message.content, str):
            parts = [message.content]
        else:
            parts = _convert_to_parts(message.content)
        messages.append({"role": role, "parts": parts})
        if i > 0:
            # Cannot have multiple messages from the same role in a row.
            if role == messages[-2]["role"]:
                raise ChatGoogleGeminiError(
                    "Cannot have multiple messages from the same role in a row."
                    " Consider merging them into a single message with multiple"
                    f" parts.\nReceived: {messages}"
                )
    return messages


def _parts_to_content(parts: List[genai.types.PartType]) -> Union[List[dict], str]:
    """Converts a list of Gemini API Part objects into a list of LangChain messages."""
    if len(parts) == 1 and parts[0].text is not None and not parts[0].inline_data:
        # Simple text response. The typical response
        return parts[0].text
    elif not parts:
        logger.warning("Gemini produced an empty response.")
        return ""
    messages = []
    for part in parts:
        if part.text is not None:
            messages.append(
                {
                    "type": "text",
                    "text": part.text,
                }
            )
        else:
            # TODO: Handle inline_data if that's a thing?
            raise ChatGoogleGeminiError(f"Unexpected part type. {part}")
    return messages


def _response_to_result(
    response: genai.types.GenerateContentResponse,
    ai_msg_t: Type[BaseMessage] = AIMessage,
    human_msg_t: Type[BaseMessage] = HumanMessage,
    chat_msg_t: Type[BaseMessage] = ChatMessage,
    generation_t: Type[ChatGeneration] = ChatGeneration,
) -> ChatResult:
    """Converts a PaLM API response into a LangChain ChatResult."""
    llm_output = {}
    if response.prompt_feedback:
        try:
            prompt_feedback = type(response.prompt_feedback).to_dict(
                response.prompt_feedback,
                use_integers_for_enums=False
            )
            llm_output["prompt_feedback"] = prompt_feedback
        except Exception as e:
            logger.debug(f"Unable to convert prompt_feedback to dict: {e}")

    generations: List[ChatGeneration] = []

    role_map = {
        "model": ai_msg_t,
        "user": human_msg_t,
    }
    for candidate in response.candidates:
        content = candidate.content
        parts_content = _parts_to_content(content.parts)
        if content.role not in role_map:
            logger.warning(
                f"Unrecognized role: {content.role}. Treating as a ChatMessage."
            )
            msg = chat_msg_t(content=parts_content, role=content.role)
        else:
            msg = role_map[content.role](content=parts_content)
        generation_info = {}
    if candidate.finish_reason:
        finish_reason = candidate.finish_reason
        # Handle both Enum and int types safely
        if hasattr(finish_reason, "name"):
            generation_info["finish_reason"] = finish_reason.name
        else:
            # Unrecognized enum value (e.g., FinishReason=12)
            generation_info["finish_reason"] = f"UNKNOWN({finish_reason})"
        if candidate.safety_ratings:
            generation_info["safety_ratings"] = [
                type(rating).to_dict(rating) for rating in candidate.safety_ratings
            ]
        generations.append(generation_t(message=msg, generation_info=generation_info))
    if not response.candidates:
        # Likely a "prompt feedback" violation (e.g., toxic input)
        # Raising an error would be different than how OpenAI handles it,
        # so we'll just log a warning and continue with an empty message.
        logger.warning(
            "Gemini produced an empty response. Continuing with empty message\n"
            f"Feedback: {response.prompt_feedback}"
        )
        generations = [generation_t(message=ai_msg_t(content=""), generation_info={})]
    return ChatResult(generations=generations, llm_output=llm_output)


class ChatGoogleGemini(BaseChatModel):
    """`Google Gemini` Chat models API.

    To use you must have the google.generativeai Python package installed and
    either:

        1. The ``GOOGLE_API_KEY``` environment variable set with your API key, or
        2. Pass your API key using the google_api_key kwarg to the ChatGoogle
           constructor.

    Example:
        .. code-block:: python

            from langchain.chat_models.google_gemini import ChatGoogleGemini
            chat = ChatGoogleGemini(model_name="gemini-pro")
            chat.invoke("Write me a ballad about LangChain")

    """

    model_name: str = Field(
        ...,
        description="""The name of the model to use.
Supported examples:
    - gemini-pro""",
    )
    max_output_tokens: int = Field(default=None, description="Max output tokens")

    client: Any  #: :meta private:
    google_api_key: Optional[str] = None
    temperature: Optional[float] = None
    """Run inference with this temperature. Must by in the closed
       interval [0.0, 1.0]."""
    top_k: Optional[int] = None
    """Decode using top-k sampling: consider the set of top_k most probable tokens.
       Must be positive."""
    n: int = 1
    """Number of chat completions to generate for each prompt. Note that the API may
       not return the full n completions if duplicates are generated."""

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"google_api_key": "GOOGLE_API_KEY"}

    @classmethod
    def is_lc_serializable(self) -> bool:
        return True

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        google_api_key = get_from_dict_or_env(
            values, "google_api_key", "GOOGLE_API_KEY"
        )
        try:
            import google.generativeai as genai

            genai.configure(api_key=google_api_key)
        except ImportError:
            raise ChatGoogleGeminiError(
                "Could not import google.generativeai python package. "
                "Please install it with `pip install google-generativeai`"
            )

        values["client"] = genai
        genai.count_text_tokens()

        if (
            values.get("temperature") is not None
            and not 0 <= values["temperature"] <= 1
        ):
            raise ValueError("temperature must be in the range [0.0, 1.0]")

        if values.get("top_p") is not None and not 0 <= values["top_p"] <= 1:
            raise ValueError("top_p must be in the range [0.0, 1.0]")

        if values.get("top_k") is not None and values["top_k"] <= 0:
            raise ValueError("top_k must be positive")
        model_name = values["model_name"]
        values["_generative_model"] = genai.GenerativeModel(model_name=model_name)
        return values

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "n": self.n,
        }

    @property
    def _generation_method(self) -> Callable:
        return self._generative_model.generate_content

    @property
    def _async_generation_method(self) -> Awaitable:
        # TODO :THIS IS BROKEN still...
        return self._generative_model.generate_content

    @property
    def _llm_type(self) -> str:
        return "google-gemini-chat"

    def _prepare_params(
        self, messages: Sequence[BaseMessage], stop: Optional[List[str]]
    ) -> Dict[str, Any]:
        contents = _messages_to_genai_contents(messages)
        gen_config = {
            k: v
            for k, v in {
                "candidate_count": self.n,
                "temperature": self.temperature,
                "stop_sequences": stop,
                "max_output_tokens": self.max_output_tokens,
            }.items()
            if v is not None
        }
        params = {
            "generation_config": gen_config,
            "contents": contents,
        }
        return params

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        params = self._prepare_params(messages, stop)
        response: genai.types.GenerateContentResponse = chat_with_retry(
            **params,
            generation_method=self._generation_method,
            **kwargs,
        )
        return _response_to_result(response)

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        return await asyncio.get_running_loop().run_in_executor(
            None, self._generate, messages, stop, run_manager, **kwargs
        )

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        params = self._prepare_params(messages, stop)
        response: genai.types.GenerateContentResponse = chat_with_retry(
            **params,
            generation_method=self._generation_method,
            **kwargs,
            stream=True,
        )
        for chunk in response:
            _chat_result = _response_to_result(
                chunk,
                ai_msg_t=AIMessageChunk,
                human_msg_t=HumanMessageChunk,
                chat_msg_t=ChatMessageChunk,
                generation_t=ChatGenerationChunk,
            )
            gen = _chat_result.generations[0]
            yield gen
            if run_manager:
                run_manager.on_llm_new_token(gen.text)
