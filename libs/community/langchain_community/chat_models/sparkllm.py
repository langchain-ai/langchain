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

SPARK_API_URL = "wss://spark-api.xf-yun.com/v3.5/chat"
SPARK_LLM_DOMAIN = "generalv3.5"


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
        return default_class(content=msg_content)  # type: ignore[call-arg]


class ChatSparkLLM(BaseChatModel):
    """IFlyTek Spark chat model integration.

    Setup:
        To use, you should have the environment variable``IFLYTEK_SPARK_API_KEY``,
        ``IFLYTEK_SPARK_API_SECRET`` and ``IFLYTEK_SPARK_APP_ID``.

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

            from langchain_community.chat_models import ChatSparkLLM

            chat = MiniMaxChat(
                api_key=api_key,
                api_secret=ak,
                model='Spark4.0 Ultra',
                # temperature=...,
                # other params...
            )

    Invoke:
        .. code-block:: python

            messages = [
                ("system", "你是一名专业的翻译家，可以将用户的中文翻译为英文。"),
                ("human", "我喜欢编程。"),
            ]
            chat.invoke(messages)

        .. code-block:: python

            AIMessage(
                content='I like programming.',
                response_metadata={
                    'token_usage': {
                        'question_tokens': 3,
                        'prompt_tokens': 16,
                        'completion_tokens': 4,
                        'total_tokens': 20
                    }
                },
                id='run-af8b3531-7bf7-47f0-bfe8-9262cb2a9d47-0'
            )

    Stream:
        .. code-block:: python

            for chunk in chat.stream(messages):
                print(chunk)

        .. code-block:: python

            content='I' id='run-fdbb57c2-2d32-4516-b894-6c5a67605d83'
            content=' like programming' id='run-fdbb57c2-2d32-4516-b894-6c5a67605d83'
            content='.' id='run-fdbb57c2-2d32-4516-b894-6c5a67605d83'

        .. code-block:: python

            stream = chat.stream(messages)
            full = next(stream)
            for chunk in stream:
                full += chunk
            full

        .. code-block:: python

            AIMessageChunk(
                content='I like programming.',
                id='run-aca2fa82-c2e4-4835-b7e2-865ddd3c46cb'
            )

    Response metadata
        .. code-block:: python

            ai_msg = chat.invoke(messages)
            ai_msg.response_metadata

        .. code-block:: python

            {
                'token_usage': {
                    'question_tokens': 3,
                    'prompt_tokens': 16,
                    'completion_tokens': 4,
                    'total_tokens': 20
                }
            }

    """  # noqa: E501

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
    spark_app_id: Optional[str] = Field(default=None, alias="app_id")
    """Automatically inferred from env var `IFLYTEK_SPARK_APP_ID` 
        if not provided."""
    spark_api_key: Optional[str] = Field(default=None, alias="api_key")
    """Automatically inferred from env var `IFLYTEK_SPARK_API_KEY` 
        if not provided."""
    spark_api_secret: Optional[str] = Field(default=None, alias="api_secret")
    """Automatically inferred from env var `IFLYTEK_SPARK_API_SECRET` 
        if not provided."""
    spark_api_url: Optional[str] = Field(default=None, alias="api_url")
    """Base URL path for API requests, leave blank if not using a proxy or service 
        emulator."""
    spark_llm_domain: Optional[str] = Field(default=None, alias="model")
    """Model name to use."""
    spark_user_id: str = "lc_user"
    streaming: bool = False
    """Whether to stream the results or not."""
    request_timeout: int = Field(30, alias="timeout")
    """request timeout for chat http requests"""
    temperature: float = Field(default=0.5)
    """What sampling temperature to use."""
    top_k: int = 4
    """What search sampling control to use."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for API call not explicitly specified."""

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

    @root_validator(pre=True)
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
            "spark_api_url",
            "IFLYTEK_SPARK_API_URL",
            SPARK_API_URL,
        )
        values["spark_llm_domain"] = get_from_dict_or_env(
            values,
            "spark_llm_domain",
            "IFLYTEK_SPARK_LLM_DOMAIN",
            SPARK_LLM_DOMAIN,
        )

        # put extra params into model_kwargs
        default_values = {
            name: field.default
            for name, field in cls.__fields__.items()
            if field.default is not None
        }
        values["model_kwargs"]["temperature"] = default_values.get("temperature")
        values["model_kwargs"]["top_k"] = default_values.get("top_k")

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
            streaming=True,
        )
        for content in self.client.subscribe(timeout=self.request_timeout):
            if "data" not in content:
                continue
            delta = content["data"]
            chunk = _convert_delta_to_message_chunk(delta, default_chunk_class)
            cg_chunk = ChatGenerationChunk(message=chunk)
            if run_manager:
                run_manager.on_llm_new_token(str(chunk.content), chunk=cg_chunk)
            yield cg_chunk

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if stream or self.streaming:
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

        self.api_url = SPARK_API_URL if not api_url else api_url
        self.app_id = app_id
        self.model_kwargs = model_kwargs
        self.spark_domain = spark_domain or SPARK_LLM_DOMAIN
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
