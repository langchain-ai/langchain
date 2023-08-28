import base64
import hashlib
import hmac
import json
import logging
import queue
import re
import ssl
import string
import threading
from datetime import datetime
from time import mktime
from typing import Any, Dict, List, Mapping, Optional
from urllib.parse import urlencode, urlparse, urlunparse
from wsgiref.handlers import format_date_time

import websocket
from _decimal import Decimal
from langchain_xfyun.callbacks.manager import (AsyncCallbackManagerForLLMRun,
                                         CallbackManagerForLLMRun)
from langchain_xfyun.chat_models.base import BaseChatModel
from langchain_xfyun.llms.utils import enforce_stop_tokens
from langchain_xfyun.schema import (AIMessage, BaseMessage, ChatGeneration,
                              ChatMessage, ChatResult, HumanMessage,
                              SystemMessage)
from langchain_xfyun.utils import get_from_dict_or_env
from pydantic import root_validator


class ChatSpark(BaseChatModel):
    r"""Wrapper around Spark's large language model.

    To use, you should pass `app_id`, `api_key`, `api_secret`
    as a named parameter to the constructor.

    Example:
        .. code-block:: python

        client = SparkLLMClient(
            app_id="<app_id>",
            api_key="<api_key>",
            api_secret="<api_secret>"
        )
    """
    client: Any = None  #: :meta private:

    max_tokens: int = 1024
    """Denotes the number of tokens to predict per generation."""

    temperature: Optional[float] = None
    """A non-negative float that tunes the degree of randomness in generation."""

    top_k: Optional[int] = None
    """Number of most likely tokens to consider at each step."""

    user_id: Optional[str] = None
    """User ID to use for the model."""

    streaming: bool = False
    """Whether to stream the results."""

    app_id: Optional[str] = None
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    api_domain: Optional[str] = None
    spark_domain: Optional[str] = None

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        values["app_id"] = get_from_dict_or_env(
            values, "app_id", "SPARK_APP_ID"
        )
        values["api_key"] = get_from_dict_or_env(
            values, "api_key", "SPARK_API_KEY"
        )
        values["api_secret"] = get_from_dict_or_env(
            values, "api_secret", "SPARK_API_SECRET"
        )

        values["client"] = SparkLLMClient(
            app_id=values["app_id"],
            api_key=values["api_key"],
            api_secret=values["api_secret"],
            api_domain=values.get('api_domain'),
            model_kwargs=values.get('spark_domain')
        )
        return values

    @property
    def _default_params(self) -> Mapping[str, Any]:
        """Get the default parameters for calling SparkLLM API."""
        d = {
            "max_tokens": self.max_tokens
        }
        if self.temperature is not None:
            d["temperature"] = self.temperature
        if self.top_k is not None:
            d["top_k"] = self.top_k
        if self.spark_domain is not None:
            d["domain"] = self.spark_domain
        return d

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {**{}, **self._default_params}

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"api_key": "API_KEY", "api_secret": "API_SECRET"}

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "spark-chat"

    @property
    def lc_serializable(self) -> bool:
        return True

    def _convert_messages_to_dicts(self, messages: List[BaseMessage]) -> list[dict]:
        """Format a list of messages into a full dict list.

        Args:
            messages (List[BaseMessage]): List of BaseMessage to combine.

        Returns:
            list[dict]
        """
        messages = messages.copy()  # don't mutate the original list

        new_messages = []
        for message in messages:
            if isinstance(message, ChatMessage):
                new_messages.append(
                    {'role': 'user', 'content': message.content})
            elif isinstance(message, HumanMessage) or isinstance(message, SystemMessage):
                new_messages.append(
                    {'role': 'user', 'content': message.content})
            elif isinstance(message, AIMessage):
                new_messages.append(
                    {'role': 'assistant', 'content': message.content})
            else:
                raise ValueError(f"Got unknown type {message}")

        return new_messages

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        messages = self._convert_messages_to_dicts(messages)

        thread = threading.Thread(target=self.client.run, args=(
            messages,
            self.user_id,
            self._default_params,
            self.streaming
        ))
        thread.start()

        completion = ""
        for content in self.client.subscribe():
            if isinstance(content, dict):
                delta = content['data']
            else:
                delta = content

            completion += delta
            if self.streaming and run_manager:
                run_manager.on_llm_new_token(
                    delta,
                )

        thread.join()
        logging.debug(f"completion: {completion}")

        if stop is not None:
            completion = enforce_stop_tokens(completion, stop)

        message = AIMessage(content=completion)
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        message = AIMessage(content='')
        return ChatResult(generations=[ChatGeneration(message=message)])

    WORD_PATTERN = re.compile(
        r'\b\w+\b|[{}]|\s'.format(re.escape(string.punctuation)))

    def get_num_tokens(self, text: str) -> float:
        """Calculate number of tokens."""
        total = Decimal(0)
        words = ChatSpark.WORD_PATTERN.findall(text)
        for word in words:
            if word:
                if '\u4e00' <= word <= '\u9fff':  # if chinese
                    total += Decimal('1.5')
                else:
                    total += Decimal('0.8')
        return int(total)


class _SparkLLMClient:
    def __init__(self, app_id: str, api_key: str, api_secret: str, api_domain: Optional[str] = None, model_kwargs: Optional[dict] = None):
        self.api_base = "wss://spark-api.xf-yun.com/v1.1/chat" if not api_domain else api_domain
        self.app_id = app_id
        self.ws_url = self.create_url(
            urlparse(self.api_base).netloc,
            urlparse(self.api_base).path,
            self.api_base,
            api_key,
            api_secret
        )
        self.model_kwargs = model_kwargs

        self.queue = queue.Queue()
        self.blocking_message = ''

    def create_url(self, host: str, path: str, api_base: str, api_key: str, api_secret: str) -> str:
        # generate timestamp by RFC1123
        date = format_date_time(mktime(datetime.now().timetuple()))

        signature_origin = f"host: {host}\ndate: {date}\nGET {path} HTTP/1.1"

        # encrypt using hmac-sha256
        signature_sha = hmac.new(api_secret.encode('utf-8'), signature_origin.encode('utf-8'),
                                 digestmod=hashlib.sha256).digest()

        signature_sha_base64 = base64.b64encode(
            signature_sha).decode(encoding='utf-8')

        authorization_origin = f'api_key="{api_key}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature_sha_base64}"'
        authorization = base64.b64encode(
            authorization_origin.encode('utf-8')).decode(encoding='utf-8')

        v = {
            "authorization": authorization,
            "date": date,
            "host": host
        }
        # generate url
        encoded_params = urlencode(v)
        parsed_url = urlparse(api_base)
        url = urlunparse((
            parsed_url.scheme,
            parsed_url.netloc,
            parsed_url.path,
            parsed_url.params,
            encoded_params,
            parsed_url.fragment
        ))
        return url

    def run(self, messages: list, user_id: str,
            model_kwargs: Optional[dict] = None, streaming: bool = False):
        websocket.enableTrace(False)
        ws = websocket.WebSocketApp(
            self.ws_url,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
            on_open=self.on_open
        )
        ws.messages = messages
        ws.user_id = user_id
        ws.model_kwargs = self.model_kwargs if model_kwargs is None else model_kwargs
        ws.streaming = streaming
        ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})

    def on_error(self, ws, error):
        self.queue.put({'error': error})
        ws.close()

    def on_close(self, ws, close_status_code, close_reason):
        self.queue.put(
            {'log': {'close_status_code': close_status_code, 'close_reason': close_reason}})
        self.queue.put({'done': True})

    def on_open(self, ws):
        self.blocking_message = ''
        data = json.dumps(self.gen_params(
            messages=ws.messages,
            user_id=ws.user_id,
            model_kwargs=ws.model_kwargs
        ))
        ws.send(data)

    def on_message(self, ws, message):
        data = json.loads(message)
        code = data['header']['code']
        if code != 0:
            self.queue.put(
                {'error': f"Code: {code}, Error: {data['header']['message']}"})
            ws.close()
        else:
            choices = data["payload"]["choices"]
            status = choices["status"]
            content = choices["text"][0]["content"]
            if ws.streaming:
                self.queue.put({'data': content})
            else:
                self.blocking_message += content

            if status == 2:
                if not ws.streaming:
                    self.queue.put({'data': self.blocking_message})
                ws.close()

    def gen_params(self, messages: list, user_id: str,
                   model_kwargs: Optional[dict] = None) -> dict:
        data = {
            "header": {
                "app_id": self.app_id,
                "uid": user_id
            },
            "parameter": {
                "chat": {
                    "domain": "general"
                }
            },
            "payload": {
                "message": {
                    "text": messages
                }
            }
        }

        if model_kwargs:
            data['parameter']['chat'].update(model_kwargs)
        logging.debug(f"Spark Request Parameters: {data}")
        return data

    def subscribe(self):
        while True:
            content = self.queue.get()
            if 'error' in content:
                raise SparkError(content['error'])
            if 'log' in content:
                logging.info(content['log'])
            if 'data' not in content:
                break
            yield content


class SparkError(Exception):
    pass
