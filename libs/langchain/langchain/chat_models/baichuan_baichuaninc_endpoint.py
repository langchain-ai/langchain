"""Baichuan chat wrapper."""
from __future__ import annotations

import requests
import json
import time
import hashlib
import logging
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
)

from langchain.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain.chat_models.base import BaseChatModel
from langchain.pydantic_v1 import Field, root_validator
from langchain.schema import ChatGeneration, ChatResult
from langchain.schema.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
)
from langchain.schema.output import ChatGenerationChunk
from langchain.utils import get_from_dict_or_env

logger = logging.getLogger(__name__)


def _convert_resp_to_message_chunk(resp: Mapping[str, Any]) -> BaseMessageChunk:
    return AIMessageChunk(
        content=resp["result"],
        role="assistant",
    )


def convert_message_to_dict(message: BaseMessage) -> dict:
    """Convert a message to a dictionary that can be passed to the API."""
    message_dict: Dict[str, Any]
    if isinstance(message, ChatMessage):
        message_dict = {"role": message.role, "content": message.content}
    elif isinstance(message, HumanMessage):
        message_dict = {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        message_dict = {"role": "assistant", "content": message.content}
        if "function_call" in message.additional_kwargs:
            message_dict["functions"] = message.additional_kwargs["function_call"]
            # If function call only, content is None not empty string
            if message_dict["content"] == "":
                message_dict["content"] = None
    elif isinstance(message, FunctionMessage):
        message_dict = {
            "role": "function",
            "content": message.content,
            "name": message.name,
        }
    else:
        raise TypeError(f"Got unknown type {message}")

    return message_dict
    

def calculate_md5(input_string):
    md5 = hashlib.md5()
    md5.update(input_string.encode('utf-8'))
    encrypted = md5.hexdigest()
    return encrypted


class BaichuanChatEndpoint():
    """
    Currently only enterprise registration is supported for use
    
    To use, you should have the environment variable ``Baichuan_AK`` and ``Baichuan_SK`` set with your
    api_key and secret_key.

    ak, sk are required parameters
    which you could get from https: // api.baichuan-ai.com

    Example:
        .. code-block: : python

        from langchain.chat_models import BaichuanChatEndpoint
        baichuan_chat = BaichuanChatEndpoint("your_ak", "your_sk","Baichuan2-13B")
        result=baichuan_chat.predict(message)
        print(result.text")
        
        Because Baichuan was no pip package made,So we will temporarily use this method and iterate and upgrade in the future

        Args:  They cannot be empty
            baichuan_ak (str): api_key
            baichuan_sk (str): secret_key
            model (str): Default Baichuan2-7B，Baichuan2-13B，Baichuan2-53B which is commercial.
            streaming (bool): Defautlt False

        Returns:
            Execute predict return response.

        """

    baichuan_ak: Optional[str] = None
    baichuan_sk: Optional[str] = None

    request_timeout: Optional[int] = 60
    """request timeout for chat http requests"""

    top_p: Optional[float] = 0.8
    temperature: Optional[float] = 0.95

    endpoint: Optional[str] = None
    """Endpoint of the Qianfan LLM, required if custom model used."""

    def __init__(self, baichuan_ak, baichuan_sk, model="Baichuan2-7B", streaming=False):
        self.baichuan_ak = baichuan_ak
        self.baichuan_sk = baichuan_sk
        self.model = "Baichuan2-7B" if model is None else model
        self.streaming = False if streaming is not None and streaming is False else True

    def predict(self, messages: List[BaseMessage]) -> Response:
        
        if self.streaming is not None and self.streaming is False:
            url = "https://api.baichuan-ai.com/v1/chat"
        elif self.streaming is not None and self.streaming is True:
            url = "https://api.baichuan-ai.com/v1/stream/chat"

        data = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": messages
                }
            ]
        }

        json_data = json.dumps(data)
        time_stamp = int(time.time())
        signature = calculate_md5(self.baichuan_sk + json_data + str(time_stamp))

        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.baichuan_ak,
            "X-BC-Request-Id": "your requestId",
            "X-BC-Timestamp": str(time_stamp),
            "X-BC-Signature": signature,
            "X-BC-Sign-Algo": "MD5",
        }
        response = requests.post(url, data=json_data, headers=headers)
        return response
