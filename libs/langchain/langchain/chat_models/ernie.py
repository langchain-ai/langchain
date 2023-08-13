import logging
import threading
from typing import Any, Dict, List, Mapping, Optional

import requests
from pydantic import root_validator

from langchain.callbacks.manager import CallbackManagerForLLMRun
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
from langchain.utils import get_from_dict_or_env

logger = logging.getLogger(__name__)


def _get_access_token(client_id: str, client_secret: str) -> str:
    base_url: str = "https://aip.baidubce.com/oauth/2.0/token"
    resp = requests.post(
        base_url,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        params={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        },
    )
    return str(resp.json().get("access_token"))


def _chat(model: str, access_token: str, json: object) -> dict:
    base_url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat"
    if model == "ERNIE-Bot-turbo":
        url = f"{base_url}/eb-instant"
    elif model == "ERNIE-Bot":
        url = f"{base_url}/completions"
    else:
        raise ValueError(f"Got unknown model_name {model}")
    resp = requests.post(
        url,
        headers={
            "Content-Type": "application/json",
        },
        params={"access_token": access_token},
        json=json,
    )
    return resp.json()


def _convert_message_to_dict(message: BaseMessage) -> dict:
    if isinstance(message, ChatMessage):
        message_dict = {"role": message.role, "content": message.content}
    elif isinstance(message, HumanMessage):
        message_dict = {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        message_dict = {"role": "assistant", "content": message.content}
    elif isinstance(message, SystemMessage):
        message_dict = {"role": "system", "content": message.content}
    else:
        raise ValueError(f"Got unknown type {message}")
    return message_dict


class ErnieChat(BaseChatModel):
    """Ernie Chat large language model.
    To use, you should have the ``client_id`` and ``client_secret`` set.
    access_token will be automatically generated based on client_id and client_secret,
    and will be regenerated after expiration
    """

    client_id: str
    client_secret: str
    access_token: str = ""
    model_name: str = "ERNIE-Bot-turbo"

    streaming: Optional[bool] = False
    top_p: Optional[float] = 0.8
    temperature: Optional[float] = 0.95
    penalty_score: Optional[float] = 1

    _lock = threading.Lock()

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        values["client_id"] = get_from_dict_or_env(
            values,
            "ernie_client_id",
            "ERNIE_CLIENT_ID",
        )
        values["client_secret"] = get_from_dict_or_env(
            values,
            "ernie_client_secret",
            "ERNIE_CLIENT_SECRET",
        )
        return values

    def _refresh_access_token_with_lock(self) -> None:
        with self._lock:
            logger.debug("Refreshing access token")
            self.access_token = _get_access_token(self.client_id, self.client_secret)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.streaming:
            raise ValueError("`streaming` option currently unsupported.")

        if not self.access_token:
            self._refresh_access_token_with_lock()
        payload = {
            "messages": [_convert_message_to_dict(m) for m in messages],
            "top_p": self.top_p,
            "temperature": self.temperature,
            "penalty_score": self.penalty_score,
        }
        resp = _chat(self.model_name, self.access_token, payload)
        if resp.get("error_code"):
            if resp.get("error_code") == 111:
                self._refresh_access_token_with_lock()
                resp = _chat(self.model_name, self.access_token, payload)
            else:
                raise ValueError(f"Error from Ernie: {resp}")
        return self._create_chat_result(resp)

    def _create_chat_result(self, response: Mapping[str, Any]) -> ChatResult:
        generations = [
            ChatGeneration(message=AIMessage(content=response.get("result")))
        ]
        token_usage = response.get("usage", {})
        llm_output = {"token_usage": token_usage, "model_name": self.model_name}
        return ChatResult(generations=generations, llm_output=llm_output)

    @property
    def _llm_type(self) -> str:
        return "ernie"
