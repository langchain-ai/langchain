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
from typing import Any, Dict, Generator, Iterator, List, Mapping, Optional, Type
from urllib.parse import urlencode, urlparse, urlunparse
from wsgiref.handlers import format_date_time

from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
)
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
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    ChatResult,
)
from langchain_core.pydantic_v1 import Field, root_validator
from langchain_core.utils import (
    get_from_dict_or_env,
    get_pydantic_field_names,
)

logger = logging.getLogger(__name__)


def _convert_message_to_dict(message: BaseMessage) -> dict:
    if isinstance(message, ChatMessage):
        message_dict = {"role": "user", "content": message.content}
    elif isinstance(message, HumanMessage):
        message_dict = {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        message_dict = {"role": "assistant", "content": message.content}
    elif isinstance(message, SystemMessage):
        message_dict = {"role": "system", "content": message.content}
    else:
        raise ValueError(f"Got unknown type {message}")

    return message_dict


def _convert_dict_to_message(_dict: Mapping[str, Any]) -> BaseMessage:
    msg_role = _dict["role"]
    msg_content = _dict["content"]
    if msg_role == "user":
        return HumanMessage(content=msg_content)
    elif msg_role == "assistant":
        content = msg_content or ""
        return AIMessage(content=content)
    elif msg_role == "system":
        return SystemMessage(content=msg_content)
    else:
        return ChatMessage(content=msg_content, role=msg_role)


def _convert_delta_to_message_chunk(
    _dict: Mapping[str, Any], default_class: Type[BaseMessageChunk]
) -> BaseMessageChunk:
    msg_role = _dict["role"]
    msg_content = _dict.get("content", "")
    if msg_role == "user" or default_class == HumanMessageChunk:
        return HumanMessageChunk(content=msg_content)
    elif msg_role == "assistant" or default_class == AIMessageChunk:
        return AIMessageChunk(content=msg_content)
    elif msg_role or default_class == ChatMessageChunk:
        return ChatMessageChunk(content=msg_content, role=msg_role)
    else:
        return default_class(content=msg_content)


class ChatSparkLLM(BaseChatModel):
    """Wrapper around iFlyTek's Spark large language model.

    To use, you should pass `app_id`, `api_key`, `api_secret`
    as a named parameter to the constructor OR set environment
    variables ``IFLYTEK_SPARK_APP_ID``, ``IFLYTEK_SPARK_API_KEY`` and
    ``IFLYTEK_SPARK_API_SECRET``

    Example:
        .. code-block:: python

        client = ChatSparkLLM(
            spark_app_id="<app_id>",
            spark_api_key="<api_key>",
            spark_api_secret="<api_secret>"
        )
    """

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this model can be serialized by Langchain."""
        return False

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {
            "spark_app_id": "IFLYTEK_SPARK_APP_ID",
            "spark_api_key": "IFLYTEK_SPARK_API_KEY",
            "spark_api_secret": "IFLYTEK_SPARK_API_SECRET",
            "spark_api_url": "IFLYTEK_SPARK_API_URL",
            "spark_llm_domain": "IFLYTEK_SPARK_LLM_DOMAIN",
        }

    client: Any = None  #: :meta private:
    spark_app_id: Optional[str] = None
    spark_api_key: Optional[str] = None
    spark_api_secret: Optional[str] = None
    spark_api_url: Optional[str] = None
    spark_llm_domain: Optional[str] = None
    spark_user_id: str = "lc_user"
    streaming: bool = False
    request_timeout: int = 30
    temperature: float = 0.5
    top_k: int = 4
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)

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

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        values["spark_app_id"] = get_from_dict_or_env(
            values,
            "spark_app_id",
            "IFLYTEK_SPARK_APP_ID",
        )
        values["spark_api_key"] = get_from_dict_or_env(
            values,
            "spark_api_key",
            "IFLYTEK_SPARK_API_KEY",
        )
        values["spark_api_secret"] = get_from_dict_or_env(
            values,
            "spark_api_secret",
            "IFLYTEK_SPARK_API_SECRET",
        )
        values["spark_app_url"] = get_from_dict_or_env(
            values,
            "spark_app_url",
            "IFLYTEK_SPARK_APP_URL",
            "wss://spark-api.xf-yun.com/v3.1/chat",
        )
        values["spark_llm_domain"] = get_from_dict_or_env(
            values,
            "spark_llm_domain",
            "IFLYTEK_SPARK_LLM_DOMAIN",
            "generalv3",
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

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        default_chunk_class = AIMessageChunk

        self.client.arun(
            [_convert_message_to_dict(m) for m in messages],
            self.spark_user_id,
            self.model_kwargs,
            self.streaming,
        )
        for content in self.client.subscribe(timeout=self.request_timeout):
            if "data" not in content:
                continue
            delta = content["data"]
            chunk = _convert_delta_to_message_chunk(delta, default_chunk_class)
            yield ChatGenerationChunk(message=chunk)
            if run_manager:
                run_manager.on_llm_new_token(str(chunk.content))

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

        self.client.arun(
            [_convert_message_to_dict(m) for m in messages],
            self.spark_user_id,
            self.model_kwargs,
            False,
        )
        completion = {}
        llm_output = {}
        for content in self.client.subscribe(timeout=self.request_timeout):
            if "usage" in content:
                llm_output["token_usage"] = content["usage"]
            if "data" not in content:
                continue
            completion = content["data"]
        message = _convert_dict_to_message(completion)
        generations = [ChatGeneration(message=message)]
        return ChatResult(generations=generations, llm_output=llm_output)

    @property
    def _llm_type(self) -> str:
        return "spark-llm-chat"


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
            "wss://spark-api.xf-yun.com/v3.1/chat" if not api_url else api_url
        )
        self.app_id = app_id
        self.ws_url = _SparkLLMClient._create_url(
            self.api_url,
            api_key,
            api_secret,
        )
        self.model_kwargs = model_kwargs
        self.spark_domain = spark_domain or "generalv3"
        self.queue: Queue[Dict] = Queue()
        self.blocking_message = {"content": "", "role": "assistant"}

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
            self.ws_url,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
            on_open=self.on_open,
        )
        ws.messages = messages
        ws.user_id = user_id
        ws.model_kwargs = self.model_kwargs if model_kwargs is None else model_kwargs
        ws.streaming = streaming
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
