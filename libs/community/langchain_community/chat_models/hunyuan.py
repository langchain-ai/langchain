import json
import logging
from typing import Any, Dict, Generator, Iterator, List, Literal, Mapping, Optional, Type

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel, generate_from_stream
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessage,
    ChatMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    SystemMessage,
    SystemMessageChunk,
)
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_core.pydantic_v1 import Field, SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env

logger = logging.getLogger(__name__)


def _convert_message_to_dict(message: BaseMessage) -> dict:
    message_dict: Dict[str, Any]
    if isinstance(message, ChatMessage):
        message_dict = {"role": message.role, "content": message.content}
    elif isinstance(message, SystemMessage):
        message_dict = {"role": "system", "content": message.content}
    elif isinstance(message, HumanMessage):
        message_dict = {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        message_dict = {"role": "assistant", "content": message.content}
    else:
        raise TypeError(f"Got unknown type {type(message)}")

    return message_dict


def _convert_dict_to_message(_dict: Mapping[str, Any]) -> BaseMessage:
    role = _dict["role"]
    if role == "system":
        return SystemMessage(content=_dict["content"])
    if role == "user":
        return HumanMessage(content=_dict["content"])
    elif role == "assistant":
        return AIMessage(content=_dict.get("content", "") or "")
    else:
        return ChatMessage(content=_dict["content"], role=role)


def _convert_delta_to_message_chunk(_dict: Mapping[str, Any], default_class: Type[BaseMessageChunk]) -> BaseMessageChunk:
    role = _dict.get("Role")
    content = _dict.get("Content") or ""

    if role == "system" or default_class == SystemMessageChunk:
        return SystemMessageChunk(content=content)
    elif role == "user" or default_class == HumanMessageChunk:
        return HumanMessageChunk(content=content)
    elif role == "assistant" or default_class == AIMessageChunk:
        return AIMessageChunk(content=content)
    elif role or default_class == ChatMessageChunk:
        return ChatMessageChunk(content=content, role=role)  # type: ignore
    else:
        return default_class(content=content)  # type: ignore


class ChatHunyuan(BaseChatModel):
    """Tencent Hunyuan chat models API by Tencent.

    For more information, see https://cloud.tencent.com/document/product/1729
    """

    tencent_cloud_secret_id: SecretStr = Field(alias="secret_id", default=None)
    """TencentCloud Secret ID"""
    tencent_cloud_secret_key: SecretStr = Field(alias="secret_key", default=None)
    """TencentCloud Secret Key"""

    model_name: str = Field(alias="model")
    """The hunyuan model name. Available: hunyuan-pro, hunyuan-standard, hunyuan-lite, hunyuan-standard-256k"""
    region: Literal["ap-guangzhou", "ap-beijing"] = "ap-guangzhou"
    """The region of hunyuan service."""
    stream_moderation: bool = False
    """Whether to enable stream moderation or not."""
    top_p: float = 0.9
    """What top p value to use."""
    temperature: float = 1.0
    """What sampling temperature to use."""

    client: Any = Field(default=None, exclude=True)
    """The tencentcloud client"""
    request_cls: Type = Field(default=None, exclude=True)
    """The request class of tencentcloud sdk"""
    message_cls: Type = Field(default=None, exclude=True)
    """The message class of tencentcloud sdk"""

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"tencent_cloud_secret_id": "TENCENT_CLOUD_SECRET_ID", "tencent_cloud_secret_key": "TENCENT_CLOUD_SECRET_KEY"}

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "chat_models", "hunyuan"]

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this model can be serialized by Langchain."""
        return True

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        values["tencent_cloud_secret_id"] = convert_to_secret_str(
            get_from_dict_or_env(
                values,
                "tencent_cloud_secret_id",
                "TENCENT_CLOUD_SECRET_ID",
            )
        )
        values["tencent_cloud_secret_key"] = convert_to_secret_str(
            get_from_dict_or_env(
                values,
                "tencent_cloud_secret_key",
                "TENCENT_CLOUD_SECRET_KEY",
            )
        )
        # Check OPENAI_ORGANIZATION for backwards compatibility.

        try:
            from tencentcloud.common.credential import Credential
            from tencentcloud.common.profile.client_profile import ClientProfile
            from tencentcloud.hunyuan.v20230901.hunyuan_client import HunyuanClient
            from tencentcloud.hunyuan.v20230901.models import ChatCompletionsRequest, Message
        except ImportError:
            raise ImportError("Could not import tencentcloud sdk python package. " "Please install it with `pip install \"tencentcloud-sdk-python>=3.0.1139\"`.")

        client_profile = ClientProfile()
        client_profile.httpProfile.pre_conn_pool_size = 3

        credential = Credential(values["tencent_cloud_secret_id"].get_secret_value(), values["tencent_cloud_secret_key"].get_secret_value())

        values["message_cls"] = Message
        values["request_cls"] = ChatCompletionsRequest
        values["client"] = HunyuanClient(credential, values["region"], client_profile)
        return values

    class Config:
        """Configuration for this pydantic object."""

        allow_population_by_field_name = True

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:

        stream_iter = self._stream(messages=messages, stop=stop, run_manager=run_manager, **kwargs)
        return generate_from_stream(stream_iter)

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        res = self._chat(messages, **kwargs)

        default_chunk_class = AIMessageChunk
        for chunk in res:
            if not isinstance(chunk, dict):
                raise ValueError(f"Error from Hunyuan api response: {chunk}")
            response = chunk.get("data", '{"error": "No data in response"}')
            response = json.loads(response)
            if "error" in response:
                raise ValueError(f"Error from Hunyuan api response: {response}")

            for choice in response["Choices"]:
                chunk = _convert_delta_to_message_chunk(choice["Delta"], default_chunk_class)
                default_chunk_class = chunk.__class__
                yield ChatGenerationChunk(message=chunk)
                if run_manager:
                    run_manager.on_llm_new_token(chunk.content)  # type: ignore

    def _chat(self, messages: List[BaseMessage], **kwargs: Any) -> Generator[Dict[str, Any], None, None]:
        _messages = [_convert_message_to_dict(m) for m in messages]

        request = self.request_cls()
        request.Stream = True
        request.Model = self.model_name
        request.Messages = []
        for _message in _messages:
            message = self.message_cls()
            message.Role = _message["role"]
            message.Content = _message["content"]
            request.Messages.append(message)

        request.Temperature = self.temperature
        request.TopP = self.top_p
        request.StreamModeration = self.stream_moderation

        res = self.client.ChatCompletions(request)
        return res  # type: ignore

    @property
    def _llm_type(self) -> str:
        return "hunyuan-chat"
