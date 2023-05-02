"""Wrapper around Google's PaLM Chat API."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from pydantic import BaseModel, root_validator

from langchain.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain.chat_models.base import BaseChatModel
from langchain.schema import (
    AIMessage,
    BaseMessage,
    ChatGeneration,
    ChatMessage,
    ChatResult,
    HumanMessage,
    SystemMessage,
)
from langchain.utils import get_from_dict_or_env

if TYPE_CHECKING:
    import google.generativeai as genai


class ChatGooglePalmError(Exception):
    pass


def _truncate_at_stop_tokens(
    text: str,
    stop: Optional[List[str]],
) -> str:
    """Truncates text at the earliest stop token found."""
    if stop is None:
        return text

    for stop_token in stop:
        stop_token_idx = text.find(stop_token)
        if stop_token_idx != -1:
            text = text[:stop_token_idx]
    return text


def _response_to_result(
    response: genai.types.ChatResponse,
    stop: Optional[List[str]],
) -> ChatResult:
    """Converts a PaLM API response into a LangChain ChatResult."""
    if not response.candidates:
        raise ChatGooglePalmError("ChatResponse must have at least one candidate.")

    generations: List[ChatGeneration] = []
    for candidate in response.candidates:
        author = candidate.get("author")
        if author is None:
            raise ChatGooglePalmError(f"ChatResponse must have an author: {candidate}")

        content = _truncate_at_stop_tokens(candidate.get("content", ""), stop)
        if content is None:
            raise ChatGooglePalmError(f"ChatResponse must have a content: {candidate}")

        if author == "ai":
            generations.append(
                ChatGeneration(text=content, message=AIMessage(content=content))
            )
        elif author == "human":
            generations.append(
                ChatGeneration(
                    text=content,
                    message=HumanMessage(content=content),
                )
            )
        else:
            generations.append(
                ChatGeneration(
                    text=content,
                    message=ChatMessage(role=author, content=content),
                )
            )

    return ChatResult(generations=generations)


def _messages_to_prompt_dict(
    input_messages: List[BaseMessage],
) -> genai.types.MessagePromptDict:
    """Converts a list of LangChain messages into a PaLM API MessagePrompt structure."""
    import google.generativeai as genai

    context: str = ""
    examples: List[genai.types.MessageDict] = []
    messages: List[genai.types.MessageDict] = []

    remaining = list(enumerate(input_messages))

    while remaining:
        index, input_message = remaining.pop(0)

        if isinstance(input_message, SystemMessage):
            if index != 0:
                raise ChatGooglePalmError("System message must be first input message.")
            context = input_message.content
        elif isinstance(input_message, HumanMessage) and input_message.example:
            if messages:
                raise ChatGooglePalmError(
                    "Message examples must come before other messages."
                )
            _, next_input_message = remaining.pop(0)
            if isinstance(next_input_message, AIMessage) and next_input_message.example:
                examples.extend(
                    [
                        genai.types.MessageDict(
                            author="human", content=input_message.content
                        ),
                        genai.types.MessageDict(
                            author="ai", content=next_input_message.content
                        ),
                    ]
                )
            else:
                raise ChatGooglePalmError(
                    "Human example message must be immediately followed by an "
                    " AI example response."
                )
        elif isinstance(input_message, AIMessage) and input_message.example:
            raise ChatGooglePalmError(
                "AI example message must be immediately preceded by a Human "
                "example message."
            )
        elif isinstance(input_message, AIMessage):
            messages.append(
                genai.types.MessageDict(author="ai", content=input_message.content)
            )
        elif isinstance(input_message, HumanMessage):
            messages.append(
                genai.types.MessageDict(author="human", content=input_message.content)
            )
        elif isinstance(input_message, ChatMessage):
            messages.append(
                genai.types.MessageDict(
                    author=input_message.role, content=input_message.content
                )
            )
        else:
            raise ChatGooglePalmError(
                "Messages without an explicit role not supported by PaLM API."
            )

    return genai.types.MessagePromptDict(
        context=context,
        examples=examples,
        messages=messages,
    )


class ChatGooglePalm(BaseChatModel, BaseModel):
    """Wrapper around Google's PaLM Chat API.

    To use you must have the google.generativeai Python package installed and
    either:

        1. The ``GOOGLE_API_KEY``` environment varaible set with your API key, or
        2. Pass your API key using the google_api_key kwarg to the ChatGoogle
           constructor.

    Example:
        .. code-block:: python

            from langchain.chat_models import ChatGooglePalm
            chat = ChatGooglePalm()

    """

    client: Any  #: :meta private:
    model_name: str = "models/chat-bison-001"
    """Model name to use."""
    google_api_key: Optional[str] = None
    temperature: Optional[float] = None
    """Run inference with this temperature. Must by in the closed
       interval [0.0, 1.0]."""
    top_p: Optional[float] = None
    """Decode using nucleus sampling: consider the smallest set of tokens whose
       probability sum is at least top_p. Must be in the closed interval [0.0, 1.0]."""
    top_k: Optional[int] = None
    """Decode using top-k sampling: consider the set of top_k most probable tokens.
       Must be positive."""
    n: int = 1
    """Number of chat completions to generate for each prompt. Note that the API may
       not return the full n completions if duplicates are generated."""

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate api key, python package exists, temperature, top_p, and top_k."""
        google_api_key = get_from_dict_or_env(
            values, "google_api_key", "GOOGLE_API_KEY"
        )
        try:
            import google.generativeai as genai

            genai.configure(api_key=google_api_key)
        except ImportError:
            raise ChatGooglePalmError(
                "Could not import google.generativeai python package."
            )

        values["client"] = genai

        if values["temperature"] is not None and not 0 <= values["temperature"] <= 1:
            raise ValueError("temperature must be in the range [0.0, 1.0]")

        if values["top_p"] is not None and not 0 <= values["top_p"] <= 1:
            raise ValueError("top_p must be in the range [0.0, 1.0]")

        if values["top_k"] is not None and values["top_k"] <= 0:
            raise ValueError("top_k must be positive")

        return values

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> ChatResult:
        prompt = _messages_to_prompt_dict(messages)

        response: genai.types.ChatResponse = self.client.chat(
            model=self.model_name,
            prompt=prompt,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            candidate_count=self.n,
        )

        return _response_to_result(response, stop)

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
    ) -> ChatResult:
        prompt = _messages_to_prompt_dict(messages)

        response: genai.types.ChatResponse = await self.client.chat_async(
            model=self.model_name,
            prompt=prompt,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            candidate_count=self.n,
        )

        return _response_to_result(response, stop)
