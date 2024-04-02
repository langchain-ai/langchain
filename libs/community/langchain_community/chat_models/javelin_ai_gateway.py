import logging
from typing import Any, Dict, List, Mapping, Optional, cast

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import (
    ChatGeneration,
    ChatResult,
)
from langchain_core.pydantic_v1 import BaseModel, Extra, SecretStr

logger = logging.getLogger(__name__)


# Ignoring type because below is valid pydantic code
# Unexpected keyword argument "extra" for "__init_subclass__" of "object"  [call-arg]
class ChatParams(BaseModel, extra=Extra.allow):
    """Parameters for the `Javelin AI Gateway` LLM."""

    temperature: float = 0.0
    stop: Optional[List[str]] = None
    max_tokens: Optional[int] = None


class ChatJavelinAIGateway(BaseChatModel):
    """`Javelin AI Gateway` chat models API.

    To use, you should have the ``javelin_sdk`` python package installed.
    For more information, see https://docs.getjavelin.io

    Example:
        .. code-block:: python

            from langchain_community.chat_models import ChatJavelinAIGateway

            chat = ChatJavelinAIGateway(
                gateway_uri="<javelin-ai-gateway-uri>",
                route="<javelin-ai-gateway-chat-route>",
                params={
                    "temperature": 0.1
                }
            )
    """

    route: str
    """The route to use for the Javelin AI Gateway API."""

    gateway_uri: Optional[str] = None
    """The URI for the Javelin AI Gateway API."""

    params: Optional[ChatParams] = None
    """Parameters for the Javelin AI Gateway LLM."""

    client: Any
    """javelin client."""

    javelin_api_key: Optional[SecretStr] = None
    """The API key for the Javelin AI Gateway."""

    def __init__(self, **kwargs: Any):
        try:
            from javelin_sdk import (
                JavelinClient,
                UnauthorizedError,
            )
        except ImportError:
            raise ImportError(
                "Could not import javelin_sdk python package. "
                "Please install it with `pip install javelin_sdk`."
            )

        super().__init__(**kwargs)
        if self.gateway_uri:
            try:
                self.client = JavelinClient(
                    base_url=self.gateway_uri,
                    api_key=cast(SecretStr, self.javelin_api_key).get_secret_value(),
                )
            except UnauthorizedError as e:
                raise ValueError("Javelin: Incorrect API Key.") from e

    @property
    def _default_params(self) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "gateway_uri": self.gateway_uri,
            "javelin_api_key": cast(SecretStr, self.javelin_api_key).get_secret_value(),
            "route": self.route,
            **(self.params.dict() if self.params else {}),
        }
        return params

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        message_dicts = [
            ChatJavelinAIGateway._convert_message_to_dict(message)
            for message in messages
        ]
        data: Dict[str, Any] = {
            "messages": message_dicts,
            **(self.params.dict() if self.params else {}),
        }

        resp = self.client.query_route(self.route, query_body=data)

        return ChatJavelinAIGateway._create_chat_result(resp.dict())

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        message_dicts = [
            ChatJavelinAIGateway._convert_message_to_dict(message)
            for message in messages
        ]
        data: Dict[str, Any] = {
            "messages": message_dicts,
            **(self.params.dict() if self.params else {}),
        }

        resp = await self.client.aquery_route(self.route, query_body=data)

        return ChatJavelinAIGateway._create_chat_result(resp.dict())

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return self._default_params

    def _get_invocation_params(
        self, stop: Optional[List[str]] = None, **kwargs: Any
    ) -> Dict[str, Any]:
        """Get the parameters used to invoke the model FOR THE CALLBACKS."""
        return {
            **self._default_params,
            **super()._get_invocation_params(stop=stop, **kwargs),
        }

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "javelin-ai-gateway-chat"

    @staticmethod
    def _convert_dict_to_message(_dict: Mapping[str, Any]) -> BaseMessage:
        role = _dict["role"]
        content = _dict["content"]
        if role == "user":
            return HumanMessage(content=content)
        elif role == "assistant":
            return AIMessage(content=content)
        elif role == "system":
            return SystemMessage(content=content)
        else:
            return ChatMessage(content=content, role=role)

    @staticmethod
    def _raise_functions_not_supported() -> None:
        raise ValueError(
            "Function messages are not supported by the Javelin AI Gateway. Please"
            " create a feature request at https://docs.getjavelin.io"
        )

    @staticmethod
    def _convert_message_to_dict(message: BaseMessage) -> dict:
        if isinstance(message, ChatMessage):
            message_dict = {"role": message.role, "content": message.content}
        elif isinstance(message, HumanMessage):
            message_dict = {"role": "user", "content": message.content}
        elif isinstance(message, AIMessage):
            message_dict = {"role": "assistant", "content": message.content}
        elif isinstance(message, SystemMessage):
            message_dict = {"role": "system", "content": message.content}
        elif isinstance(message, FunctionMessage):
            raise ValueError(
                "Function messages are not supported by the Javelin AI Gateway. Please"
                " create a feature request at https://docs.getjavelin.io"
            )
        else:
            raise ValueError(f"Got unknown message type: {message}")

        if "function_call" in message.additional_kwargs:
            ChatJavelinAIGateway._raise_functions_not_supported()
        if message.additional_kwargs:
            logger.warning(
                "Additional message arguments are unsupported by Javelin AI Gateway "
                " and will be ignored: %s",
                message.additional_kwargs,
            )
        return message_dict

    @staticmethod
    def _create_chat_result(response: Mapping[str, Any]) -> ChatResult:
        generations = []
        for candidate in response["llm_response"]["choices"]:
            message = ChatJavelinAIGateway._convert_dict_to_message(
                candidate["message"]
            )
            message_metadata = candidate.get("metadata", {})
            gen = ChatGeneration(
                message=message,
                generation_info=dict(message_metadata),
            )
            generations.append(gen)

        response_metadata = response.get("metadata", {})
        return ChatResult(generations=generations, llm_output=response_metadata)
