import logging
import threading
from typing import Any, Dict, List, Mapping, Optional

import requests

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.chat_models.base import BaseChatModel
from langchain.pydantic_v1 import root_validator
from langchain.schema import (
    AIMessage,
    BaseMessage,
    ChatGeneration,
    ChatMessage,
    ChatResult,
    HumanMessage,
)
from langchain.utils import get_from_dict_or_env

logger = logging.getLogger(__name__)


def _convert_message_to_dict(message: BaseMessage) -> dict:
    if isinstance(message, ChatMessage):
        message_dict = {"role": message.role, "content": message.content}
    elif isinstance(message, HumanMessage):
        message_dict = {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        message_dict = {"role": "assistant", "content": message.content}
    else:
        raise ValueError(f"Got unknown type {message}")
    return message_dict


class ErnieBotChat(BaseChatModel):
    """`ERNIE-Bot` large language model.

    ERNIE-Bot is a large language model developed by Baidu,
    covering a huge amount of Chinese data.

    To use, you should have the `ernie_client_id` and `ernie_client_secret` set,
    or set the environment variable `ERNIE_CLIENT_ID` and `ERNIE_CLIENT_SECRET`.

    Note:
    access_token will be automatically generated based on client_id and client_secret,
    and will be regenerated after expiration (30 days).

    Default model is `ERNIE-Bot-turbo`,
    currently supported models are `ERNIE-Bot-turbo`, `ERNIE-Bot`

    Example:
        .. code-block:: python

            from langchain.chat_models import ErnieBotChat
            chat = ErnieBotChat(model_name='ERNIE-Bot')

    """

    ernie_client_id: Optional[str] = None
    ernie_client_secret: Optional[str] = None
    access_token: Optional[str] = None

    model_name: str = "ERNIE-Bot-turbo"

    streaming: Optional[bool] = False
    top_p: Optional[float] = 0.8
    temperature: Optional[float] = 0.95
    penalty_score: Optional[float] = 1

    _lock = threading.Lock()

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        values["ernie_client_id"] = get_from_dict_or_env(
            values,
            "ernie_client_id",
            "ERNIE_CLIENT_ID",
        )
        values["ernie_client_secret"] = get_from_dict_or_env(
            values,
            "ernie_client_secret",
            "ERNIE_CLIENT_SECRET",
        )
        return values

    def _chat(self, payload: object) -> dict:
        base_url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat"
        if self.model_name == "ERNIE-Bot-turbo":
            url = f"{base_url}/eb-instant"
        elif self.model_name == "ERNIE-Bot":
            url = f"{base_url}/completions"
        else:
            raise ValueError(f"Got unknown model_name {self.model_name}")
        resp = requests.post(
            url,
            headers={
                "Content-Type": "application/json",
            },
            params={"access_token": self.access_token},
            json=payload,
        )
        return resp.json()

    def _refresh_access_token_with_lock(self) -> None:
        with self._lock:
            logger.debug("Refreshing access token")
            base_url: str = "https://aip.baidubce.com/oauth/2.0/token"
            resp = requests.post(
                base_url,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                params={
                    "grant_type": "client_credentials",
                    "client_id": self.ernie_client_id,
                    "client_secret": self.ernie_client_secret,
                },
            )
            self.access_token = str(resp.json().get("access_token"))

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
            **kwargs,
        }
        logger.debug(f"Payload for ernie api is {payload}")
        resp = self._chat(payload)
        if resp.get("error_code"):
            if resp.get("error_code") == 111:
                logger.debug("access_token expired, refresh it")
                self._refresh_access_token_with_lock()
                resp = self._chat(payload)
            else:
                raise ValueError(f"Error from ErnieChat api response: {resp}")
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
        return "ernie-bot-chat"
