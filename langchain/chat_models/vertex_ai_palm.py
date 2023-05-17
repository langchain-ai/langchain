"""Wrapper around Google Cloud Platform Vertex AI PaLM Chat API."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional

import logging

from pydantic import BaseModel, root_validator

from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_exponential,
)


from langchain.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain.chat_models.base import BaseChatModel
from langchain.schema import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    ChatResult,
    HumanMessage,
    SystemMessage,
)
from langchain.utils import get_from_dict_or_env

logger = logging.getLogger(__name__)

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

def _messages_to_prompt_dict(
    input_messages: List[BaseMessage],
) -> dict:
    """Converts a list of LangChain messages into a PaLM API-compatible structure."""
    from vertexai.preview.language_models import InputOutputTextPair
    context: str = ""
    examples: List[dict] = []
    history: List[tuple] = []
    prompt: str = ""

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
                examples.append(InputOutputTextPair(input_message.content, next_input_message.content))
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
        elif isinstance(input_message, HumanMessage):
            _, next_input_message = remaining.pop(0)
            if isinstance(next_input_message, AIMessage):
                history.append(InputOutputTextPair(input_message.content, next_input_message.content))
            else:
                raise ChatGooglePalmError(
                    "Human historical message must be immediately followed by an "
                    " AI historical response."
                )
        elif isinstance(input_message, HumanMessage):
            prompt = input_message.content
        elif isinstance(input_message, AIMessage) or isinstance(input_message, ChatMessage):
            raise ChatGooglePalmError("vertexai.preview.langugagemodel.ChatModel.start_chat.message expects a user message as input")

    return {
        "context": context,
        "examples": examples,
        "history": history,
        "prompt": prompt
    }

class ChatVertexAIGooglePalm(BaseChatModel, BaseModel):
    """Wrapper around Google Cloud's Vertex AI PaLM Chat API.

    To use you must have the google-cloud-aiplatform Python package installed and
    either:

        1. Have credentials configured for your enviornment (gcloud, workload identity, etc...)
        2. Pass your service account key json using the google_apu kwarg to the ChatGoogle
           constructor.

        *see: https://cloud.google.com/docs/authentication/application-default-credentials#GAC

    Example:
        .. code-block:: python

            from langchain.chat_models import ChatGoogleVertexAIPalm
            chat = ChatVertexAIGooglePalm()

    """

    client: Any  #: :meta private:
    google_application_credentials: Optional[str]
    model_name: str = "models/text-bison@001"
    """Model name to use."""
    temperature: float = 0.2
    """Run inference with this temperature. Must by in the closed interval
       [0.0, 1.0]."""
    top_p: Optional[float] = 0.8
    """Decode using nucleus sampling: consider the smallest set of tokens whose
       probability sum is at least top_p. Must be in the closed interval [0.0, 1.0]."""
    top_k: Optional[int] = 40
    """Decode using top-k sampling: consider the set of top_k most probable tokens.
       Must be positive."""
    max_output_tokens: Optional[int] = 256
    """Maximum number of tokens to include in a candidate. Must be greater than zero.
       If unset, will default to 256."""

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate api key, python package exists."""
        google_auth = get_from_dict_or_env(
            values, "google_application_credentials", "GOOGLE_APPLICATION_CREDENTIALS"
        )
        try:
            from vertexai.preview.language_models import ChatModel

        except ImportError:
            raise ImportError("Could not import vertexai python package. Try running `pip install google-cloud-aiplatform>=1.25.0`")
    
        if values["temperature"] is not None and not 0 <= values["temperature"] <= 1:
            raise ValueError("temperature must be in the range [0.0, 1.0]")

        if values["top_p"] is not None and not 0 <= values["top_p"] <= 1:
            raise ValueError("top_p must be in the range [0.0, 1.0]")

        if values["top_k"] is not None and values["top_k"] <= 0:
            raise ValueError("top_k must be positive")
        
        model =  ChatModel.from_pretrained(values["model_name"])
        values["client"] = model.start_chat(
            max_output_tokens=values["max_output_tokens"],
            temperature = values["temperature"],
            top_k = values["top_k"],
            top_p = values["top_p"]
            )

        return values

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> ChatResult:
        prompt_dict = _messages_to_prompt_dict(messages)

        self.client._history = prompt_dict["history"]
        self.client._context = prompt_dict["context"]
        self.client._examples = prompt_dict["examples"]
        prompt = prompt_dict["prompt"]

        completion_with_retry = retry(
                reraise=True,
                stop=stop_after_attempt(3),
                wait=wait_exponential(
                    multiplier=1,
                    min=4,
                    max=10
                ),
                before_sleep=before_sleep_log(logger, logging.WARNING),
            )(self.client.predict)
        response = completion_with_retry(
            prompt,
            self.max_output_tokens,
            self.temperature,
            self.top_k,
            self.top_p
        )

        return _truncate_at_stop_tokens(response.text, stop)

    async def _agenerate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
    ) -> ChatResult:
        raise NotImplementedError()


    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_output_tokens": self.max_output_tokens,
        }

    @property
    def _llm_type(self) -> str:
        return "google-palm-chat"
