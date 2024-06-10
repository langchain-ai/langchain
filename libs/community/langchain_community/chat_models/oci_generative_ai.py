from typing import Any, Dict, Iterator, List, Optional, Sequence
from abc import ABC, abstractmethod
import json

from langchain_core.language_models.chat_models import BaseChatModel, generate_from_stream
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, ChatMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_community.llms.utils import enforce_stop_tokens

from langchain_core.callbacks import CallbackManagerForLLMRun

from langchain_core.pydantic_v1 import Extra
from langchain_community.llms.oci_generative_ai import OCIGenAIBase


# oci 2.127 new message roles caps, new sys role, llamaindex incompatability
# stream issues: (1) command-r stream hangs at end with stop (2) stream cohere command does not remove stop
# test with dedicated cluster
# command-r tools ?
class Provider(ABC):
    @property
    @abstractmethod
    def stop_sequence_key(self) -> str:
        ...

    @abstractmethod
    def chat_response_to_text(self, response: Any) -> str:
        ...

    @abstractmethod
    def chat_stream_to_text(self, event_data: Dict) -> str:
        ...

    @abstractmethod
    def chat_generation_info(self, response: Any) -> Dict[str, Any]:
        ...

    @abstractmethod
    def get_role(self, message: BaseMessage) -> str:
        ...
        
    @abstractmethod
    def messages_to_oci_params(self, messages: Sequence[ChatMessage]) -> Dict[str, Any]:
        ...

class CohereProvider(Provider):
    stop_sequence_key = "stop_sequences"

    def __init__(self) -> None:
        try:
            from oci.generative_ai_inference import models
            
        except ImportError as ex:
            raise ModuleNotFoundError(
                "Could not import oci python package. "
                "Please make sure you have the oci package installed."
            ) from ex

        self.oci_chat_request = models.CohereChatRequest
        self.oci_chat_message = {"USER": models.CohereUserMessage, "CHATBOT": models.CohereChatBotMessage, "SYSTEM": models.CohereSystemMessage}
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

        oci_chat_history = [self.oci_chat_message[self.get_role(msg)](message=msg.content) for msg in messages[:-1]]
        #oci_chat_history = [self.oci_chat_message(role=self.get_role(msg), message=msg.content) for msg in messages[:-1]]
        oci_params = {
            "message": messages[-1].content,
            "chat_history": oci_chat_history,
            "api_format": self.chat_api_format
        }
    
        return oci_params
    
class MetaProvider(Provider):
    stop_sequence_key = "stop"
    
    def __init__(self) -> None:
        try:
            from oci.generative_ai_inference import models
            
        except ImportError as ex:
            raise ModuleNotFoundError(
                "Could not import oci python package. "
                "Please make sure you have the oci package installed."
            ) from ex

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
            "time_created": str(response.data.chat_response.time_created)
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

        oci_messages = [self.oci_chat_message[self.get_role(msg)](content=[self.oci_chat_message_content(text=msg.content)]) for msg in messages]
        oci_params = {
            "messages": oci_messages,
            "api_format": self.chat_api_format,
            "top_k": -1
        }
    
        return oci_params

PROVIDERS = {
    "cohere": CohereProvider(),
    "meta": MetaProvider(),
}

CUSTOM_ENDPOINT_PREFIX = "ocid1.generativeaiendpoint"

class ChatOCIGenAI(BaseChatModel, OCIGenAIBase):
    """OCI large language chat models.

    To authenticate, the OCI client uses the methods described in
    https://docs.oracle.com/en-us/iaas/Content/API/Concepts/sdk_authentication_methods.htm

    The authentifcation method is passed through auth_type and should be one of:
    API_KEY (default), SECURITY_TOKEN, INSTANCE_PRINCIPAL, RESOURCE_PRINCIPAL

    Make sure you have the required policies (profile/roles) to
    access the OCI Generative AI service.
    If a specific config profile is used, you must pass
    the name of the profile (from ~/.oci/config) through auth_profile.

    To use, you must provide the compartment id
    along with the endpoint url, and model id
    as named parameters to the constructor.

    Example:
        .. code-block:: python

            from langchain_community.chat_models import OCIGenAIChat

            llm = OCIGenAIChat(
                    model_id="MY_MODEL_ID",
                    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
                    compartment_id="MY_OCID"
                )
    """

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "oci_generative_ai_chat"

    def _prepare_request(
        self, messages: List[BaseMessage], stop: Optional[List[str]], kwargs: Dict[str, Any], provider: Provider, stream: bool
    ) -> Dict[str, Any]:
        try:
            from oci.generative_ai_inference import models
            
        except ImportError as ex:
            raise ModuleNotFoundError(
                "Could not import oci python package. "
                "Please make sure you have the oci package installed."
            ) from ex

        oci_params = provider.messages_to_oci_params(messages)
        oci_params["is_stream"] = stream # self.is_stream
        _model_kwargs = self.model_kwargs or {}

        if stop is not None:
            _model_kwargs[provider.stop_sequence_key] = stop

        chat_params = {**_model_kwargs, **kwargs, **oci_params}

        if self.model_id.startswith(CUSTOM_ENDPOINT_PREFIX):
            serving_mode = models.DedicatedServingMode(endpoint_id=self.model_id)
        else:
            serving_mode = models.OnDemandServingMode(model_id=self.model_id)

        request = models.ChatDetails(
            compartment_id=self.compartment_id,
            serving_mode=serving_mode,
            chat_request=provider.oci_chat_request(**chat_params),
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
            stream_iter = self._stream(messages, stop=stop, run_manager=run_manager, **kwargs)
            return generate_from_stream(stream_iter)

        provider = PROVIDERS[self._get_provider()]
        request = self._prepare_request(messages, stop, kwargs, provider, stream=False)
        response = self.client.chat(request)
        
        content = provider.chat_response_to_text(response)

        if stop is not None:
            content = enforce_stop_tokens(content, stop)

        generation_info = provider.chat_generation_info(response)

        llm_output = {
            "model_id": response.data.model_id,
            "model_version": response.data.model_version,
            "request_id": response.request_id,
            "content-length": response.headers["content-length"]
            }

        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=content), generation_info=generation_info)],
            llm_output=llm_output
        )

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        
        provider = PROVIDERS[self._get_provider()]
        request = self._prepare_request(messages, stop, kwargs, provider, stream=True)
        response = self.client.chat(request)
                
        for event in response.data.events():
            delta = provider.chat_stream_to_text(json.loads(event.data))
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=delta))
            if run_manager:
                run_manager.on_llm_new_token(delta, chunk=chunk)
            yield chunk

        