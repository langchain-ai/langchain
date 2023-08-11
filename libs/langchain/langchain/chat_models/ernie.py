import logging
import threading
from typing import Optional, List, Any, Mapping

import requests
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.chat_models.base import BaseChatModel
from langchain.schema import BaseMessage, ChatResult, ChatMessage, HumanMessage, AIMessage, SystemMessage, \
    ChatGeneration

logger = logging.getLogger(__name__)


def _get_access_token(client_id: str, client_secret: str) -> str:
    base_url: str = "https://aip.baidubce.com/oauth/2.0/token"
    resp = requests.post(base_url, headers={
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }, params={
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret
    })
    return str(resp.json().get("access_token"))


def _chat(model: str, access_token: str, json: object) -> dict:
    if model == "ERNIE-Bot-turbo":
        base_url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/eb-instant"
    elif model == "ERNIE-Bot":
        base_url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions"
    else:
        raise ValueError(f"Got unknown model_name {model}")
    resp = requests.post(base_url, headers={
        'Content-Type': 'application/json',
    }, params={
        "access_token": access_token
    }, json=json, )
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
    client_id: str
    client_secret: str
    access_token: Optional[str]
    model_name: str = "ERNIE-Bot-turbo"

    streaming: Optional[bool] = False
    top_p: Optional[float] = 0.8
    temperature: Optional[float] = 0.95
    penalty_score: Optional[float] = 1

    _lock = threading.Lock()

    def _refresh_access_token_with_lock(self):
        with self._lock:
            logger.debug("Refreshing access token")
            self.access_token = _get_access_token(self.client_id, self.client_secret)

    def _generate(self,
                  messages: List[BaseMessage],
                  stop: Optional[List[str]] = None,
                  run_manager: Optional[CallbackManagerForLLMRun] = None,
                  **kwargs: Any) -> ChatResult:
        if self.streaming:
            raise ValueError("`streaming` option currently unsupported.")

        if not self.access_token:
            self._refresh_access_token_with_lock()
        payload = {
            "messages": [_convert_message_to_dict(m) for m in messages],
            "top_p": self.top_p,
            "temperature": self.temperature,
            "penalty_score": self.penalty_score
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
        generations = [ChatGeneration(message=AIMessage(content=response.get("result")))]
        token_usage = response.get("usage", {})
        llm_output = {"token_usage": token_usage, "model_name": self.model_name}
        return ChatResult(generations=generations, llm_output=llm_output)

    @property
    def _llm_type(self) -> str:
        return "ernie"
