import json
import logging
from typing import Any, Dict, List, Mapping, Optional

import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.pydantic_v1 import root_validator
from langchain_core.utils import get_pydantic_field_names

logger = logging.getLogger(__name__)


def _convert_dict_to_message(_dict: Mapping[str, Any]) -> BaseMessage:
    role = _dict["role"]
    if role == "user":
        return HumanMessage(content=_dict["content"])
    elif role == "assistant":
        return AIMessage(content=_dict.get("content", "") or "")
    else:
        return ChatMessage(content=_dict["content"], role=role)


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
        raise TypeError(f"Got unknown type {message}")

    return message_dict


class WasmChatService(BaseChatModel):
    """Chat with LLMs via `llama-api-server`

    For the information about `llama-api-server`, visit https://github.com/second-state/llama-utils
    """

    request_timeout: int = 60
    """request timeout for chat http requests"""
    service_url: Optional[str] = None
    """URL of WasmChat service"""
    model: str = "NA"
    """model name, default is `NA`."""

    class Config:
        """Configuration for this pydantic object."""

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

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        res = self._chat(messages, **kwargs)

        if res.status_code != 200:
            raise ValueError(f"Error code: {res.status_code}, reason: {res.reason}")

        response = res.json()

        return self._create_chat_result(response)

    def _chat(self, messages: List[BaseMessage], **kwargs: Any) -> requests.Response:
        if self.service_url is None:
            res = requests.models.Response()
            res.status_code = 503
            res.reason = "The IP address or port of the chat service is incorrect."
            return res

        service_url = f"{self.service_url}/v1/chat/completions"

        payload = {
            "model": self.model,
            "messages": [_convert_message_to_dict(m) for m in messages],
        }

        res = requests.post(
            url=service_url,
            timeout=self.request_timeout,
            headers={
                "accept": "application/json",
                "Content-Type": "application/json",
            },
            data=json.dumps(payload),
        )

        return res

    def _create_chat_result(self, response: Mapping[str, Any]) -> ChatResult:
        message = _convert_dict_to_message(response["choices"][0].get("message"))
        generations = [ChatGeneration(message=message)]

        token_usage = response["usage"]
        llm_output = {"token_usage": token_usage, "model": self.model}
        return ChatResult(generations=generations, llm_output=llm_output)

    @property
    def _llm_type(self) -> str:
        return "wasm-chat"
