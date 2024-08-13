from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import queue
import threading
from datetime import datetime
from queue import Queue
from time import mktime
from typing import Any, Dict, Generator, Iterator, List, Optional
from urllib.parse import urlencode, urlparse, urlunparse
from wsgiref.handlers import format_date_time

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from langchain_core.pydantic_v1 import Field
from langchain_core.utils import get_from_dict_or_env, pre_init

logger = logging.getLogger(__name__)


class SparkLLM(LLM):
    """iFlyTek Spark completion model integration.

    Setup:
        To use, you should set environment variables ``IFLYTEK_SPARK_APP_ID``,
        ``IFLYTEK_SPARK_API_KEY`` and ``IFLYTEK_SPARK_API_SECRET``.

    .. code-block:: bash

            export IFLYTEK_SPARK_APP_ID="your-app-id"
            export IFLYTEK_SPARK_API_KEY="your-api-key"
            export IFLYTEK_SPARK_API_SECRET="your-api-secret"

    Key init args — completion params:
        model: Optional[str]
            Name of IFLYTEK SPARK model to use.
        temperature: Optional[float]
            Sampling temperature.
        top_k: Optional[float]
            What search sampling control to use.
        streaming: Optional[bool]
             Whether to stream the results or not.

    Key init args — client params:
        app_id: Optional[str]
            IFLYTEK SPARK API KEY. Automatically inferred from env var `IFLYTEK_SPARK_APP_ID` if not provided.
        api_key: Optional[str]
            IFLYTEK SPARK API KEY. If not passed in will be read from env var IFLYTEK_SPARK_API_KEY.
        api_secret: Optional[str]
            IFLYTEK SPARK API SECRET. If not passed in will be read from env var IFLYTEK_SPARK_API_SECRET.
        api_url: Optional[str]
            Base URL for API requests.
        timeout: Optional[int]
            Timeout for requests.

    See full list of supported init args and their descriptions in the params section.

    Instantiate:
        .. code-block:: python

            from langchain_community.llms import SparkLLM

            llm = SparkLLM(
                app_id="your-app-id",
                api_key="your-api_key",
                api_secret="your-api-secret",
                # model='Spark4.0 Ultra',
                # temperature=...,
                # other params...
            )

    Invoke:
        .. code-block:: python

            input_text = "用50个字左右阐述，生命的意义在于"
            llm.invoke(input_text)

        .. code-block:: python

            '生命的意义在于实现自我价值，追求内心的平静与快乐，同时为他人和社会带来正面影响。'

    Stream:
        .. code-block:: python

            for chunk in llm.stream(input_text):
                print(chunk)

        .. code-block:: python

            生命 | 的意义在于 | 不断探索和 | 实现个人潜能，通过 | 学习 | 、成长和对社会 | 的贡献，追求内心的满足和幸福。

    Async:
        .. code-block:: python

            await llm.ainvoke(input_text)

            # stream:
            # async for chunk in llm.astream(input_text):
            #    print(chunk)

            # batch:
            # await llm.abatch([input_text])

        .. code-block:: python

            '生命的意义在于实现自我价值，追求内心的平静与快乐，同时为他人和社会带来正面影响。'

    """  # noqa: E501

    client: Any = None  #: :meta private:
    spark_app_id: Optional[str] = Field(default=None, alias="app_id")
    """Automatically inferred from env var `IFLYTEK_SPARK_APP_ID` 
        if not provided."""
    spark_api_key: Optional[str] = Field(default=None, alias="api_key")
    """IFLYTEK SPARK API KEY. If not passed in will be read from 
        env var IFLYTEK_SPARK_API_KEY."""
    spark_api_secret: Optional[str] = Field(default=None, alias="api_secret")
    """IFLYTEK SPARK API SECRET. If not passed in will be read from 
        env var IFLYTEK_SPARK_API_SECRET."""
    spark_api_url: Optional[str] = Field(default=None, alias="api_url")
    """Base URL path for API requests, leave blank if not using a proxy or service 
        emulator."""
    spark_llm_domain: Optional[str] = Field(default=None, alias="model")
    """Model name to use."""
    spark_user_id: str = "lc_user"
    streaming: bool = False
    """Whether to stream the results or not."""
    request_timeout: int = Field(default=30, alias="timeout")
    """request timeout for chat http requests"""
    temperature: float = 0.5
    """What sampling temperature to use."""
    top_k: int = 4
    """What search sampling control to use."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for API call not explicitly specified."""

    @pre_init
    def validate_environment(cls, values: Dict) -> Dict:
        values["spark_app_id"] = get_from_dict_or_env(
            values,
            ["spark_app_id", "app_id"],
            "IFLYTEK_SPARK_APP_ID",
        )
        values["spark_api_key"] = get_from_dict_or_env(
            values,
            ["spark_api_key", "api_key"],
            "IFLYTEK_SPARK_API_KEY",
        )
        values["spark_api_secret"] = get_from_dict_or_env(
            values,
            ["spark_api_secret", "api_secret"],
            "IFLYTEK_SPARK_API_SECRET",
        )
        values["spark_api_url"] = get_from_dict_or_env(
            values,
            ["spark_api_url", "api_url"],
            "IFLYTEK_SPARK_API_URL",
            "wss://spark-api.xf-yun.com/v3.5/chat",
        )
        values["spark_llm_domain"] = get_from_dict_or_env(
            values,
            ["spark_llm_domain", "model"],
            "IFLYTEK_SPARK_LLM_DOMAIN",
            "generalv3.5",
        )
        # put extra params into model_kwargs
        values["model_kwargs"]["temperature"] = values["temperature"] or cls.temperature
        values["model_kwargs"]["top_k"] = values["top_k"] or cls.top_k

        values["client"] = _SparkLLMClient(
            app_id=values["spark_app_id"],
            api_key=values["spark_api_key"],
            api_secret=values["spark_api_secret"],
            api_url=values["spark_api_url"],
            spark_domain=values["spark_llm_domain"],
            model_kwargs=values["model_kwargs"],
        )
        return values

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "spark-llm-chat"

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling SparkLLM API."""
        normal_params = {
            "spark_llm_domain": self.spark_llm_domain,
            "stream": self.streaming,
            "request_timeout": self.request_timeout,
            "top_k": self.top_k,
            "temperature": self.temperature,
        }

        return {**normal_params, **self.model_kwargs}

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call out to an sparkllm for each generation with a prompt.
        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.
        Returns:
            The string generated by the llm.

        Example:
            .. code-block:: python
                response = client("Tell me a joke.")
        """
        if self.streaming:
            completion = ""
            for chunk in self._stream(prompt, stop, run_manager, **kwargs):
                completion += chunk.text
            return completion
        completion = ""
        self.client.arun(
            [{"role": "user", "content": prompt}],
            self.spark_user_id,
            self.model_kwargs,
            self.streaming,
        )
        for content in self.client.subscribe(timeout=self.request_timeout):
            if "data" not in content:
                continue
            completion = content["data"]["content"]

        return completion

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        self.client.run(
            [{"role": "user", "content": prompt}],
            self.spark_user_id,
            self.model_kwargs,
            True,
        )
        for content in self.client.subscribe(timeout=self.request_timeout):
            if "data" not in content:
                continue
            delta = content["data"]
            if run_manager:
                run_manager.on_llm_new_token(delta)
            yield GenerationChunk(text=delta["content"])


class _SparkLLMClient:
    """
    Use websocket-client to call the SparkLLM interface provided by Xfyun,
    which is the iFlyTek's open platform for AI capabilities
    """

    def __init__(
        self,
        app_id: str,
        api_key: str,
        api_secret: str,
        api_url: Optional[str] = None,
        spark_domain: Optional[str] = None,
        model_kwargs: Optional[dict] = None,
    ):
        try:
            import websocket

            self.websocket_client = websocket
        except ImportError:
            raise ImportError(
                "Could not import websocket client python package. "
                "Please install it with `pip install websocket-client`."
            )

        self.api_url = (
            "wss://spark-api.xf-yun.com/v3.5/chat" if not api_url else api_url
        )
        self.app_id = app_id
        self.model_kwargs = model_kwargs
        self.spark_domain = spark_domain or "generalv3.5"
        self.queue: Queue[Dict] = Queue()
        self.blocking_message = {"content": "", "role": "assistant"}
        self.api_key = api_key
        self.api_secret = api_secret

    @staticmethod
    def _create_url(api_url: str, api_key: str, api_secret: str) -> str:
        """
        Generate a request url with an api key and an api secret.
        """
        # generate timestamp by RFC1123
        date = format_date_time(mktime(datetime.now().timetuple()))

        # urlparse
        parsed_url = urlparse(api_url)
        host = parsed_url.netloc
        path = parsed_url.path

        signature_origin = f"host: {host}\ndate: {date}\nGET {path} HTTP/1.1"

        # encrypt using hmac-sha256
        signature_sha = hmac.new(
            api_secret.encode("utf-8"),
            signature_origin.encode("utf-8"),
            digestmod=hashlib.sha256,
        ).digest()

        signature_sha_base64 = base64.b64encode(signature_sha).decode(encoding="utf-8")

        authorization_origin = f'api_key="{api_key}", algorithm="hmac-sha256", \
        headers="host date request-line", signature="{signature_sha_base64}"'
        authorization = base64.b64encode(authorization_origin.encode("utf-8")).decode(
            encoding="utf-8"
        )

        # generate url
        params_dict = {"authorization": authorization, "date": date, "host": host}
        encoded_params = urlencode(params_dict)
        url = urlunparse(
            (
                parsed_url.scheme,
                parsed_url.netloc,
                parsed_url.path,
                parsed_url.params,
                encoded_params,
                parsed_url.fragment,
            )
        )
        return url

    def run(
        self,
        messages: List[Dict],
        user_id: str,
        model_kwargs: Optional[dict] = None,
        streaming: bool = False,
    ) -> None:
        self.websocket_client.enableTrace(False)
        ws = self.websocket_client.WebSocketApp(
            _SparkLLMClient._create_url(
                self.api_url,
                self.api_key,
                self.api_secret,
            ),
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
            on_open=self.on_open,
        )
        ws.messages = messages  # type: ignore[attr-defined]
        ws.user_id = user_id  # type: ignore[attr-defined]
        ws.model_kwargs = self.model_kwargs if model_kwargs is None else model_kwargs  # type: ignore[attr-defined]
        ws.streaming = streaming  # type: ignore[attr-defined]
        ws.run_forever()

    def arun(
        self,
        messages: List[Dict],
        user_id: str,
        model_kwargs: Optional[dict] = None,
        streaming: bool = False,
    ) -> threading.Thread:
        ws_thread = threading.Thread(
            target=self.run,
            args=(
                messages,
                user_id,
                model_kwargs,
                streaming,
            ),
        )
        ws_thread.start()
        return ws_thread

    def on_error(self, ws: Any, error: Optional[Any]) -> None:
        self.queue.put({"error": error})
        ws.close()

    def on_close(self, ws: Any, close_status_code: int, close_reason: str) -> None:
        logger.debug(
            {
                "log": {
                    "close_status_code": close_status_code,
                    "close_reason": close_reason,
                }
            }
        )
        self.queue.put({"done": True})

    def on_open(self, ws: Any) -> None:
        self.blocking_message = {"content": "", "role": "assistant"}
        data = json.dumps(
            self.gen_params(
                messages=ws.messages, user_id=ws.user_id, model_kwargs=ws.model_kwargs
            )
        )
        ws.send(data)

    def on_message(self, ws: Any, message: str) -> None:
        data = json.loads(message)
        code = data["header"]["code"]
        if code != 0:
            self.queue.put(
                {"error": f"Code: {code}, Error: {data['header']['message']}"}
            )
            ws.close()
        else:
            choices = data["payload"]["choices"]
            status = choices["status"]
            content = choices["text"][0]["content"]
            if ws.streaming:
                self.queue.put({"data": choices["text"][0]})
            else:
                self.blocking_message["content"] += content
            if status == 2:
                if not ws.streaming:
                    self.queue.put({"data": self.blocking_message})
                usage_data = (
                    data.get("payload", {}).get("usage", {}).get("text", {})
                    if data
                    else {}
                )
                self.queue.put({"usage": usage_data})
                ws.close()

    def gen_params(
        self, messages: list, user_id: str, model_kwargs: Optional[dict] = None
    ) -> dict:
        data: Dict = {
            "header": {"app_id": self.app_id, "uid": user_id},
            "parameter": {"chat": {"domain": self.spark_domain}},
            "payload": {"message": {"text": messages}},
        }

        if model_kwargs:
            data["parameter"]["chat"].update(model_kwargs)
        logger.debug(f"Spark Request Parameters: {data}")
        return data

    def subscribe(self, timeout: Optional[int] = 30) -> Generator[Dict, None, None]:
        while True:
            try:
                content = self.queue.get(timeout=timeout)
            except queue.Empty as _:
                raise TimeoutError(
                    f"SparkLLMClient wait LLM api response timeout {timeout} seconds"
                )
            if "error" in content:
                raise ConnectionError(content["error"])
            if "usage" in content:
                yield content
                continue
            if "done" in content:
                break
            if "data" not in content:
                break
            yield content
