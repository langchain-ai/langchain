import json
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import (
    BaseChatModel,
    generate_from_stream,
)
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.pydantic_v1 import Extra

from langchain_community.llms.oci_generative_ai import OCIGenAIBase
from langchain_community.llms.utils import enforce_stop_tokens

CUSTOM_ENDPOINT_PREFIX = "ocid1.generativeaiendpoint"


class Provider(ABC):
    @property
    @abstractmethod
    def stop_sequence_key(self) -> str: ...

    @abstractmethod
    def chat_response_to_text(self, response: Any) -> str: ...

    @abstractmethod
    def chat_stream_to_text(self, event_data: Dict) -> str: ...

    @abstractmethod
    def chat_generation_info(self, response: Any) -> Dict[str, Any]: ...

    @abstractmethod
    def get_role(self, message: BaseMessage) -> str: ...

    @abstractmethod
    def messages_to_oci_params(self, messages: Any) -> Dict[str, Any]: ...


class CohereProvider(Provider):
    stop_sequence_key = "stop_sequences"

    def __init__(self) -> None:
        from oci.generative_ai_inference import models

        self.oci_chat_request = models.CohereChatRequest
        self.oci_chat_message = {
            "USER": models.CohereUserMessage,
            "CHATBOT": models.CohereChatBotMessage,
            "SYSTEM": models.CohereSystemMessage,
        }
        self.chat_api_format = models.BaseChatRequest.API_FORMAT_COHERE

    def chat_response_to_text(self, response: Any) -> str:
        return response.data.chat_response.text

    def chat_stream_to_text(self, event_data: Dict) -> str:
        if "text" in event_data and "finishReason" not in event_data:
            return event_data["text"]
        else:
            return ""

    def chat_generation_info(self, response: Any) -> Dict[str, Any]:
        return {
            "finish_reason": response.data.chat_response.finish_reason,
        }

    def get_role(self, message: BaseMessage) -> str:
        if isinstance(message, HumanMessage):
            return "USER"
        elif isinstance(message, AIMessage):
            return "CHATBOT"
        elif isinstance(message, SystemMessage):
            return "SYSTEM"
        else:
            raise ValueError(f"Got unknown type {message}")

    def messages_to_oci_params(self, messages: Sequence[ChatMessage]) -> Dict[str, Any]:
        oci_chat_history = [
            self.oci_chat_message[self.get_role(msg)](message=msg.content)
            for msg in messages[:-1]
        ]
        oci_params = {
            "message": messages[-1].content,
            "chat_history": oci_chat_history,
            "api_format": self.chat_api_format,
        }

        return oci_params


class MetaProvider(Provider):
    stop_sequence_key = "stop"

    def __init__(self) -> None:
        from oci.generative_ai_inference import models

        self.oci_chat_request = models.GenericChatRequest
        self.oci_chat_message = {
            "USER": models.UserMessage,
            "SYSTEM": models.SystemMessage,
            "ASSISTANT": models.AssistantMessage,
        }
        self.oci_chat_message_content = models.TextContent
        self.chat_api_format = models.BaseChatRequest.API_FORMAT_GENERIC

    def chat_response_to_text(self, response: Any) -> str:
        return response.data.chat_response.choices[0].message.content[0].text

    def chat_stream_to_text(self, event_data: Dict) -> str:
        if "message" in event_data:
            return event_data["message"]["content"][0]["text"]
        else:
            return ""

    def chat_generation_info(self, response: Any) -> Dict[str, Any]:
        return {
            "finish_reason": response.data.chat_response.choices[0].finish_reason,
            "time_created": str(response.data.chat_response.time_created),
        }

    def get_role(self, message: BaseMessage) -> str:
        # meta only supports alternating user/assistant roles
        if isinstance(message, HumanMessage):
            return "USER"
        elif isinstance(message, AIMessage):
            return "ASSISTANT"
        elif isinstance(message, SystemMessage):
            return "SYSTEM"
        else:
            raise ValueError(f"Got unknown type {message}")

    def messages_to_oci_params(self, messages: List[BaseMessage]) -> Dict[str, Any]:
        oci_messages = [
            self.oci_chat_message[self.get_role(msg)](
                content=[self.oci_chat_message_content(text=msg.content)]
            )
            for msg in messages
        ]
        oci_params = {
            "messages": oci_messages,
            "api_format": self.chat_api_format,
            "top_k": -1,
        }

        return oci_params


class ChatOCIGenAI(BaseChatModel, OCIGenAIBase):
    """ChatOCIGenAI chat model integration.

    Setup:
      Install ``langchain-community`` and the ``oci`` sdk.

      .. code-block:: bash

          pip install -U langchain-community oci

    Key init args — completion params:
        model_id: str
            Id of the OCIGenAI chat model to use, e.g., cohere.command-r-16k.
        is_stream: bool
            Whether to stream back partial progress
        model_kwargs: Optional[Dict]
            Keyword arguments to pass to the specific model used, e.g., temperature, max_tokens.

    Key init args — client params:
        service_endpoint: str
            The endpoint URL for the OCIGenAI service, e.g., https://inference.generativeai.us-chicago-1.oci.oraclecloud.com.
        compartment_id: str
            The compartment OCID.
        auth_type: str
            The authentication type to use, e.g., API_KEY (default), SECURITY_TOKEN, INSTANCE_PRINCIPAL, RESOURCE_PRINCIPAL.
        auth_profile: Optional[str]
            The name of the profile in ~/.oci/config, if not specified , DEFAULT will be used.
        provider: str
            Provider name of the model. Default to None, will try to be derived from the model_id otherwise, requires user input.
    See full list of supported init args and their descriptions in the params section.

    Instantiate:
        .. code-block:: python

            from langchain_community.chat_models import ChatOCIGenAI

            chat = ChatOCIGenAI(
                model_id="cohere.command-r-16k",
                service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
                compartment_id="MY_OCID",
                model_kwargs={"temperature": 0.7, "max_tokens": 500},
            )

    Invoke:
        .. code-block:: python
            messages = [
                SystemMessage(content="your are an AI assistant."),
                AIMessage(content="Hi there human!"),
                HumanMessage(content="tell me a joke."),
            ]
            response = chat.invoke(messages)

    Stream:
        .. code-block:: python

        for r in chat.stream(messages):
            print(r.content, end="", flush=True)

    Response metadata
        .. code-block:: python

        response = chat.invoke(messages)
        print(response.response_metadata)

    """  # noqa: E501

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "oci_generative_ai_chat"

    @property
    def _provider_map(self) -> Mapping[str, Any]:
        """Get the provider map"""
        return {
            "cohere": CohereProvider(),
            "meta": MetaProvider(),
        }

    @property
    def _provider(self) -> Any:
        """Get the internal provider object"""
        return self._get_provider(provider_map=self._provider_map)

    def _prepare_request(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]],
        kwargs: Dict[str, Any],
        stream: bool,
    ) -> Dict[str, Any]:
        try:
            from oci.generative_ai_inference import models

        except ImportError as ex:
            raise ModuleNotFoundError(
                "Could not import oci python package. "
                "Please make sure you have the oci package installed."
            ) from ex
        oci_params = self._provider.messages_to_oci_params(messages)
        oci_params["is_stream"] = stream  # self.is_stream
        _model_kwargs = self.model_kwargs or {}

        if stop is not None:
            _model_kwargs[self._provider.stop_sequence_key] = stop

        chat_params = {**_model_kwargs, **kwargs, **oci_params}

        if self.model_id.startswith(CUSTOM_ENDPOINT_PREFIX):
            serving_mode = models.DedicatedServingMode(endpoint_id=self.model_id)
        else:
            serving_mode = models.OnDemandServingMode(model_id=self.model_id)

        request = models.ChatDetails(
            compartment_id=self.compartment_id,
            serving_mode=serving_mode,
            chat_request=self._provider.oci_chat_request(**chat_params),
        )

        return request

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Call out to a OCIGenAI chat model.

        Args:
            messages: list of LangChain messages
            stop: Optional list of stop words to use.

        Returns:
            LangChain ChatResult

        Example:
            .. code-block:: python

               messages = [
                            HumanMessage(content="hello!"),
                            AIMessage(content="Hi there human!"),
                            HumanMessage(content="Meow!")
                          ]

               response = llm.invoke(messages)
        """
        if self.is_stream:
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)

        request = self._prepare_request(messages, stop, kwargs, stream=False)
        response = self.client.chat(request)

        content = self._provider.chat_response_to_text(response)

        if stop is not None:
            content = enforce_stop_tokens(content, stop)

        generation_info = self._provider.chat_generation_info(response)

        llm_output = {
            "model_id": response.data.model_id,
            "model_version": response.data.model_version,
            "request_id": response.request_id,
            "content-length": response.headers["content-length"],
        }

        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(content=content), generation_info=generation_info
                )
            ],
            llm_output=llm_output,
        )

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        request = self._prepare_request(messages, stop, kwargs, stream=True)
        response = self.client.chat(request)

        for event in response.data.events():
            delta = self._provider.chat_stream_to_text(json.loads(event.data))
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=delta))
            if run_manager:
                run_manager.on_llm_new_token(delta, chunk=chunk)
            yield chunk
