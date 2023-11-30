import asyncio
import logging
from functools import partial
from typing import Any, Dict, List, Mapping, Optional
from urllib.parse import urlparse

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.pydantic_v1 import (
    Field,
    PrivateAttr,
    root_validator,
)
from langchain_core.utils import get_pydantic_field_names

logger = logging.getLogger(__name__)


class ChatMlflow(BaseChatModel):
    """`MLflow` chat models API.

    To use, you should have the `mlflow[genai]` python package installed.
    For more information, see https://mlflow.org/docs/latest/llms/deployments/server.html.

    Example:
        .. code-block:: python

            from langchain.chat_models import ChatMlflow

            chat = ChatMlflow(
                target_uri="http://localhost:5000",
                endpoint="chat",
                temperature-0.1,
            )
    """

    endpoint: str
    """The endpoint to use."""
    target_uri: str
    """The target URI to use."""
    temperature: float = 0.0
    """The sampling temperature."""
    n: int = 1
    """The number of completion choices to generate."""
    stop: Optional[List[str]] = None
    """The stop sequence."""
    max_tokens: Optional[int] = None
    """The maximum number of tokens to generate."""
    extra_params: dict = Field(default_factory=dict)
    """Any extra parameters to pass to the endpoint."""
    _client: Any = PrivateAttr()

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._validate_uri()
        try:
            from mlflow.deployments import get_deploy_client

            self._client = get_deploy_client(self.target_uri)
        except ImportError as e:
            raise ImportError(
                "Failed to create the client. "
                f"Please run `pip install mlflow{self._mlflow_extras}` to install "
                "required dependencies."
            ) from e

    @property
    def _mlflow_extras(self) -> str:
        return "[genai]"

    def _validate_uri(self) -> None:
        if self.target_uri == "databricks":
            return
        allowed = ["http", "https", "databricks"]
        if urlparse(self.target_uri).scheme not in allowed:
            raise ValueError(
                f"Invalid target URI: {self.target_uri}. "
                f"The scheme must be one of {allowed}."
            )

    @root_validator(pre=True)
    def build_extra(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = get_pydantic_field_names(cls)
        extra = values.get("extra_params", {})
        for field_name in list(values):
            if field_name in extra:
                raise ValueError(f"Found {field_name} supplied twice.")
            if field_name not in all_required_field_names:
                extra[field_name] = values.pop(field_name)

        invalid_model_kwargs = all_required_field_names.intersection(extra.keys())
        if invalid_model_kwargs:
            raise ValueError(
                f"Parameters {invalid_model_kwargs} should be specified explicitly. "
                f"Instead they were passed in as part of `extra_params` parameter."
            )

        values["extra_params"] = extra
        return values

    @property
    def _default_params(self) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "target_uri": self.target_uri,
            "endpoint": self.endpoint,
            "temperature": self.temperature,
            "n": self.n,
            "stop": self.stop,
            "max_tokens": self.max_tokens,
            **self.extra_params,
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
            ChatMlflow._convert_message_to_dict(message) for message in messages
        ]
        data: Dict[str, Any] = {
            "messages": message_dicts,
            "temperature": self.temperature,
            "n": self.n,
            "stop": stop or self.stop,
            "max_tokens": self.max_tokens,
            **self.extra_params,
        }

        resp = self._client.predict(endpoint=self.endpoint, inputs=data)
        return ChatMlflow._create_chat_result(resp)

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
        return "mlflow-chat"

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
            "Function messages are not supported by Databricks. Please"
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
                "Function messages are not supported by Databricks. Please"
                " create a feature request at https://github.com/mlflow/mlflow/issues."
            )
        else:
            raise ValueError(f"Got unknown message type: {message}")

        if "function_call" in message.additional_kwargs:
            ChatMlflow._raise_functions_not_supported()
        if message.additional_kwargs:
            logger.warning(
                "Additional message arguments are unsupported by Databricks"
                " and will be ignored: %s",
                message.additional_kwargs,
            )
        return message_dict

    @staticmethod
    def _create_chat_result(response: Mapping[str, Any]) -> ChatResult:
        generations = []
        for choice in response["choices"]:
            message = ChatMlflow._convert_dict_to_message(choice["message"])
            usage = choice.get("usage", {})
            gen = ChatGeneration(
                message=message,
                generation_info=usage,
            )
            generations.append(gen)

        usage = response.get("usage", {})
        return ChatResult(generations=generations, llm_output=usage)
