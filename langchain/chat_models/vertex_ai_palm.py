"""Wrapper around Google Cloud Platform Vertex AI PaLM Chat API."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional

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
    ChatGeneration,
    ChatMessage,
    ChatResult,
    HumanMessage,
    SystemMessage,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import vertexai.preview.language_models as palm


class ChatGoogleVertexAIPalmError(Exception):
    pass


def _response_to_result(
    response: palm.TextGenerationResponse,
    prompt: str,
    stop: Optional[List[str]],
) -> ChatResult:
    """Converts a PaLM API response into a LangChain ChatResult."""

    prediction = response._prediction_response.predictions[0]

    if not prediction.get("candidates"):
        raise ChatGoogleVertexAIPalmError(
            "ChatResponse must have at least one candidate."
        )

    generations: List[ChatGeneration] = []

    # vertex ai doesn't return the history in the prediction call.
    # So we just add it in here.
    generations.append(
        ChatGeneration(
            text=prompt,
            message=HumanMessage(content=prompt),
        )
    )
    for candidate in prediction.get("candidates"):
        author = candidate.get("author")
        if author is None:
            raise ChatGoogleVertexAIPalmError(
                f"ChatResponse must have an author: {candidate}"
            )

        content = _truncate_at_stop_tokens(candidate.get("content", ""), stop)
        if content is None:
            raise ChatGoogleVertexAIPalmError(
                f"ChatResponse must have a content: {candidate}"
            )

        if author == "1":
            generations.append(
                ChatGeneration(text=content, message=AIMessage(content=content))
            )
        else:
            generations.append(
                ChatGeneration(
                    text=content,
                    message=ChatMessage(role=author, content=content),
                )
            )

    return ChatResult(generations=generations)


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
                raise ChatGoogleVertexAIPalmError(
                    "System message must be first input message."
                )
            context = input_message.content
        elif isinstance(input_message, HumanMessage) and input_message.example:
            if prompt:
                raise ChatGoogleVertexAIPalmError(
                    "Message examples must come before other messages."
                )
            _, next_input_message = remaining.pop(0)
            if isinstance(next_input_message, AIMessage) and next_input_message.example:
                examples.append(
                    InputOutputTextPair(
                        input_message.content, next_input_message.content
                    )
                )
            else:
                raise ChatGoogleVertexAIPalmError(
                    "Human example message must be immediately followed by an "
                    " AI example response."
                )
        elif isinstance(input_message, AIMessage) and input_message.example:
            raise ChatGoogleVertexAIPalmError(
                "AI example message must be immediately preceded by a Human "
                "example message."
            )
        elif isinstance(input_message, HumanMessage) and remaining:
            _, next_input_message = remaining.pop(0)
            if isinstance(next_input_message, AIMessage):
                history.append(
                    InputOutputTextPair(
                        input_message.content, next_input_message.content
                    )
                )
            else:
                raise ChatGoogleVertexAIPalmError(
                    "Human historical message must be immediately followed by an "
                    " AI historical response."
                )
        elif isinstance(input_message, HumanMessage):
            prompt = input_message.content
        elif isinstance(input_message, AIMessage) or isinstance(
            input_message, ChatMessage
        ):
            raise ChatGoogleVertexAIPalmError(
                "vertexai.preview.langugagemodel.ChatModel."
                "start_chat.message expects a user message as input"
            )

    return {
        "context": context,
        "examples": examples,
        "history": history,
        "prompt": prompt,
    }


class ChatGoogleCloudVertexAIPalm(BaseChatModel, BaseModel):
    """Wrapper around Google Cloud's Vertex AI PaLM Chat API.

    To use you must have the google-cloud-aiplatform Python package installed and
    either:

    1. Have credentials configured for your environment -
        (gcloud, workload identity, etc...)
    2. Store the path to a service account JSON file as
        the GOOGLE_APPLICATION_CREDENTIALS environment variable

    Example:
        .. code-block:: python

            from langchain.chat_models import ChatGoogleVertexAIPalm
            chat = ChatGoogleCloudVertexAIPalm()

    """

    client: Any  #: :meta private:
    model_name: str = "chat-bison@001"
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
        """Validate auth and that python package exists."""

        import google.auth

        credentials, project_id = google.auth.default()

        try:
            from vertexai.preview.language_models import ChatModel

        except ImportError:
            raise ImportError(
                "Could not import vertexai python package."
                "Try running `pip install google-cloud-aiplatform>=1.25.0`"
            )

        if values["temperature"] is not None and not 0 <= values["temperature"] <= 1:
            raise ValueError("temperature must be in the range [0.0, 1.0]")

        if values["top_p"] is not None and not 0 <= values["top_p"] <= 1:
            raise ValueError("top_p must be in the range [0.0, 1.0]")

        if values["top_k"] is not None and values["top_k"] <= 0:
            raise ValueError("top_k must be positive")

        model = ChatModel.from_pretrained(values["model_name"])
        values["client"] = model.start_chat(
            max_output_tokens=values["max_output_tokens"],
            temperature=values["temperature"],
            top_k=values["top_k"],
            top_p=values["top_p"],
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
            wait=wait_exponential(multiplier=1, min=4, max=10),
            before_sleep=before_sleep_log(logger, logging.WARNING),
        )(self.client.send_message)
        response = completion_with_retry(
            prompt,
            max_output_tokens=self.max_output_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
        )

        return _response_to_result(response, prompt, stop)

    async def _agenerate(
        self,
        prompts: List[BaseMessage],
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
        return "google_cloud_vertex_ai_palm_chat"
