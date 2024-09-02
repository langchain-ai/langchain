import json
import logging
from typing import Any, Dict, Iterator, List, Mapping, Optional, Type

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import (
    BaseChatModel,
    generate_from_stream,
)
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
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.pydantic_v1 import Field, SecretStr, root_validator
from langchain_core.utils import (
    convert_to_secret_str,
    get_from_dict_or_env,
    get_pydantic_field_names,
    pre_init,
)

logger = logging.getLogger(__name__)


def _convert_message_to_dict(message: BaseMessage) -> dict:
    message_dict: Dict[str, Any]
    if isinstance(message, ChatMessage):
        message_dict = {"Role": message.role, "Content": message.content}
    elif isinstance(message, SystemMessage):
        message_dict = {"Role": "system", "Content": message.content}
    elif isinstance(message, HumanMessage):
        message_dict = {"Role": "user", "Content": message.content}
    elif isinstance(message, AIMessage):
        message_dict = {"Role": "assistant", "Content": message.content}
    else:
        raise TypeError(f"Got unknown type {message}")

    return message_dict


def _convert_dict_to_message(_dict: Mapping[str, Any]) -> BaseMessage:
    role = _dict["Role"]
    if role == "system":
        return SystemMessage(content=_dict.get("Content", "") or "")
    elif role == "user":
        return HumanMessage(content=_dict["Content"])
    elif role == "assistant":
        return AIMessage(content=_dict.get("Content", "") or "")
    else:
        return ChatMessage(content=_dict["Content"], role=role)


def _convert_delta_to_message_chunk(
    _dict: Mapping[str, Any], default_class: Type[BaseMessageChunk]
) -> BaseMessageChunk:
    role = _dict.get("Role")
    content = _dict.get("Content") or ""

    if role == "user" or default_class == HumanMessageChunk:
        return HumanMessageChunk(content=content)
    elif role == "assistant" or default_class == AIMessageChunk:
        return AIMessageChunk(content=content)
    elif role or default_class == ChatMessageChunk:
        return ChatMessageChunk(content=content, role=role)  # type: ignore[arg-type]
    else:
        return default_class(content=content)  # type: ignore[call-arg]


def _create_chat_result(response: Mapping[str, Any]) -> ChatResult:
    generations = []
    for choice in response["Choices"]:
        message = _convert_dict_to_message(choice["Message"])
        message.id = response.get("Id", "")
        generations.append(ChatGeneration(message=message))

    token_usage = response["Usage"]
    llm_output = {"token_usage": token_usage}
    return ChatResult(generations=generations, llm_output=llm_output)


class ChatHunyuan(BaseChatModel):
    """Tencent Hunyuan chat models API by Tencent.

    For more information, see https://cloud.tencent.com/document/product/1729
    """

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {
            "hunyuan_app_id": "HUNYUAN_APP_ID",
            "hunyuan_secret_id": "HUNYUAN_SECRET_ID",
            "hunyuan_secret_key": "HUNYUAN_SECRET_KEY",
        }

    @property
    def lc_serializable(self) -> bool:
        return True

    hunyuan_app_id: Optional[int] = None
    """Hunyuan App ID"""
    hunyuan_secret_id: Optional[str] = None
    """Hunyuan Secret ID"""
    hunyuan_secret_key: Optional[SecretStr] = None
    """Hunyuan Secret Key"""
    streaming: bool = False
    """Whether to stream the results or not."""
    request_timeout: int = 60
    """Timeout for requests to Hunyuan API. Default is 60 seconds."""
    temperature: float = 1.0
    """What sampling temperature to use."""
    top_p: float = 1.0
    """What probability mass to use."""
    model: str = "hunyuan-lite"
    """What Model to use.
     Optional model:
    - hunyuan-lite
    - hunyuan-standard
    - hunyuan-standard-256K
    - hunyuan-pro
    - hunyuan-code
    - hunyuan-role
    - hunyuan-functioncall
    - hunyuan-vision
    """
    stream_moderation: bool = False
    """Whether to review the results or not when streaming is true."""
    enable_enhancement: bool = True
    """Whether to enhancement the results or not."""

    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for API call not explicitly specified."""

    class Config:
        allow_population_by_field_name = True

    @root_validator(pre=True)
    def build_extra(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = get_pydantic_field_names(cls)
        extra = values.get("model_kwargs", {})
        for field_name in list(values):
            if field_name in extra:
                raise ValueError(f"Found {field_name} supplied twice.")
            if field_name not in all_required_field_names:
                logger.warning(
                    f"""WARNING! {field_name} is not default parameter.
                    {field_name} was transferred to model_kwargs.
                    Please confirm that {field_name} is what you intended."""
                )
                extra[field_name] = values.pop(field_name)

        invalid_model_kwargs = all_required_field_names.intersection(extra.keys())
        if invalid_model_kwargs:
            raise ValueError(
                f"Parameters {invalid_model_kwargs} should be specified explicitly. "
                f"Instead they were passed in as part of `model_kwargs` parameter."
            )

        values["model_kwargs"] = extra
        return values

    @pre_init
    def validate_environment(cls, values: Dict) -> Dict:
        values["hunyuan_app_id"] = get_from_dict_or_env(
            values,
            "hunyuan_app_id",
            "HUNYUAN_APP_ID",
        )
        values["hunyuan_secret_id"] = get_from_dict_or_env(
            values,
            "hunyuan_secret_id",
            "HUNYUAN_SECRET_ID",
        )
        values["hunyuan_secret_key"] = convert_to_secret_str(
            get_from_dict_or_env(
                values,
                "hunyuan_secret_key",
                "HUNYUAN_SECRET_KEY",
            )
        )
        return values

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling Hunyuan API."""
        normal_params = {
            "Temperature": self.temperature,
            "TopP": self.top_p,
            "Model": self.model,
            "Stream": self.streaming,
            "StreamModeration": self.stream_moderation,
            "EnableEnhancement": self.enable_enhancement,
        }
        return {**normal_params, **self.model_kwargs}

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.streaming:
            stream_iter = self._stream(
                messages=messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)

        res = self._chat(messages, **kwargs)
        return _create_chat_result(json.loads(res.to_json_string()))

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
            chunk = chunk.get("data", "")
            if len(chunk) == 0:
                continue
            response = json.loads(chunk)
            if "error" in response:
                raise ValueError(f"Error from Hunyuan api response: {response}")

            for choice in response["Choices"]:
                chunk = _convert_delta_to_message_chunk(
                    choice["Delta"], default_chunk_class
                )
                chunk.id = response.get("Id", "")
                default_chunk_class = chunk.__class__
                cg_chunk = ChatGenerationChunk(message=chunk)
                if run_manager:
                    run_manager.on_llm_new_token(chunk.content, chunk=cg_chunk)
                yield cg_chunk

    def _chat(self, messages: List[BaseMessage], **kwargs: Any) -> Any:
        if self.hunyuan_secret_key is None:
            raise ValueError("Hunyuan secret key is not set.")

        try:
            from tencentcloud.common import credential
            from tencentcloud.hunyuan.v20230901 import hunyuan_client, models
        except ImportError:
            raise ImportError(
                "Could not import tencentcloud python package. "
                "Please install it with `pip install tencentcloud-sdk-python`."
            )

        parameters = {**self._default_params, **kwargs}
        cred = credential.Credential(
            self.hunyuan_secret_id, str(self.hunyuan_secret_key.get_secret_value())
        )
        client = hunyuan_client.HunyuanClient(cred, "")
        req = models.ChatCompletionsRequest()
        params = {
            "Messages": [_convert_message_to_dict(m) for m in messages],
            **parameters,
        }
        req.from_json_string(json.dumps(params))
        resp = client.ChatCompletions(req)
        return resp

    @property
    def _llm_type(self) -> str:
        return "hunyuan-chat"
