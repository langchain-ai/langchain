import asyncio
import logging
import warnings
from functools import partial
from typing import Any, Dict, List, Mapping, Optional

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
from langchain_core.pydantic_v1 import BaseModel, Extra

from langchain.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain.chat_models.base import BaseChatModel

logger = logging.getLogger(__name__)


# Ignoring type because below is valid pydantic code
# Unexpected keyword argument "extra" for "__init_subclass__" of "object"  [call-arg]
class ChatParams(BaseModel, extra=Extra.allow):  # type: ignore[call-arg]
    """Parameters for the `MLflow AI Gateway` LLM."""

    temperature: float = 0.0
    candidate_count: int = 1
    """The number of candidates to return."""
    stop: Optional[List[str]] = None
    max_tokens: Optional[int] = None


class ChatMLflowAIGateway(BaseChatModel):
    """`MLflow AI Gateway` chat models API.

    To use, you should have the ``mlflow[gateway]`` python package installed.
    For more information, see https://mlflow.org/docs/latest/gateway/index.html.

    Example:
        .. code-block:: python

            from langchain.chat_models import ChatMLflowAIGateway

            chat = ChatMLflowAIGateway(
                gateway_uri="<your-mlflow-ai-gateway-uri>",
                route="<your-mlflow-ai-gateway-chat-route>",
                params={
                    "temperature": 0.1
                }
            )
    """

    def __init__(self, **kwargs: Any):
        warnings.warn(
            "`ChatMLflowAIGateway` is deprecated. Use `ChatMlflow` or "
            "`ChatDatabricks` instead.",
            DeprecationWarning,
        )
        try:
            import mlflow.gateway
        except ImportError as e:
            raise ImportError(
                "Could not import `mlflow.gateway` module. "
                "Please install it with `pip install mlflow[gateway]`."
            ) from e

        super().__init__(**kwargs)
        if self.gateway_uri:
            mlflow.gateway.set_gateway_uri(self.gateway_uri)

    route: str
    gateway_uri: Optional[str] = None
    params: Optional[ChatParams] = None

    @property
    def _default_params(self) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "gateway_uri": self.gateway_uri,
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
        try:
            import mlflow.gateway
        except ImportError as e:
            raise ImportError(
                "Could not import `mlflow.gateway` module. "
                "Please install it with `pip install mlflow[gateway]`."
            ) from e

        message_dicts = [
            ChatMLflowAIGateway._convert_message_to_dict(message)
            for message in messages
        ]
        data: Dict[str, Any] = {
            "messages": message_dicts,
            **(self.params.dict() if self.params else {}),
        }

        resp = mlflow.gateway.query(self.route, data=data)
        return ChatMLflowAIGateway._create_chat_result(resp)

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        func = partial(
            self._generate, messages, stop=stop, run_manager=run_manager, **kwargs
        )
        return await asyncio.get_event_loop().run_in_executor(None, func)

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
        return "mlflow-ai-gateway-chat"

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
            "Function messages are not supported by the MLflow AI Gateway. Please"
            " create a feature request at https://github.com/mlflow/mlflow/issues."
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
                "Function messages are not supported by the MLflow AI Gateway. Please"
                " create a feature request at https://github.com/mlflow/mlflow/issues."
            )
        else:
            raise ValueError(f"Got unknown message type: {message}")

        if "function_call" in message.additional_kwargs:
            ChatMLflowAIGateway._raise_functions_not_supported()
        if message.additional_kwargs:
            logger.warning(
                "Additional message arguments are unsupported by MLflow AI Gateway "
                " and will be ignored: %s",
                message.additional_kwargs,
            )
        return message_dict

    @staticmethod
    def _create_chat_result(response: Mapping[str, Any]) -> ChatResult:
        generations = []
        for candidate in response["candidates"]:
            message = ChatMLflowAIGateway._convert_dict_to_message(candidate["message"])
            message_metadata = candidate.get("metadata", {})
            gen = ChatGeneration(
                message=message,
                generation_info=dict(message_metadata),
            )
            generations.append(gen)

        response_metadata = response.get("metadata", {})
        return ChatResult(generations=generations, llm_output=response_metadata)
