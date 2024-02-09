from __future__ import annotations

import json
import logging
import time
from functools import partial
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Generator,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import aiohttp
import requests
from langchain_core.pydantic_v1 import (
    BaseModel,
    Field,
    PrivateAttr,
    SecretStr,
    root_validator,
)
from langchain_core.utils import get_from_dict_or_env
from requests.models import Response

logger = logging.getLogger(__name__)


class NVEModel(BaseModel):

    """
    Underlying Client for interacting with the AI Foundation Model Function API.
    Leveraged by the NVIDIABaseModel to provide a simple requests-oriented interface.
    Direct abstraction over NGC-recommended streaming/non-streaming Python solutions.

    NOTE: Models in the playground does not currently support raw text continuation.
    """

    ## Core defaults. These probably should not be changed
    fetch_url_format: str = Field("https://api.nvcf.nvidia.com/v2/nvcf/pexec/status/")
    call_invoke_base: str = Field("https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions")
    func_list_format: str = Field("https://api.nvcf.nvidia.com/v2/nvcf/functions")
    get_session_fn: Callable = Field(requests.Session)
    get_asession_fn: Callable = Field(aiohttp.ClientSession)

    nvidia_api_key: SecretStr = Field(
        ...,
        description="API key for NVIDIA Foundation Endpoints. Starts with `nvapi-`",
    )
    is_staging: bool = Field(False, description="Whether to use staging API")

    ## Generation arguments
    timeout: float = Field(60, ge=0, description="Timeout for waiting on response (s)")
    interval: float = Field(0.02, ge=0, description="Interval for pulling response")
    last_inputs: dict = Field({}, description="Last inputs sent over to the server")
    payload_fn: Callable = Field(lambda d: d, description="Function to process payload")
    headers_tmpl: dict = Field(
        ...,
        description="Headers template for API calls."
        " Should contain `call` and `stream` keys.",
    )
    _available_functions: Optional[List[dict]] = PrivateAttr(default=None)
    _available_models: Optional[dict] = PrivateAttr(default=None)

    @property
    def headers(self) -> dict:
        """Return headers with API key injected"""
        headers_ = self.headers_tmpl.copy()
        for header in headers_.values():
            if "{nvidia_api_key}" in header["Authorization"]:
                header["Authorization"] = header["Authorization"].format(
                    nvidia_api_key=self.nvidia_api_key.get_secret_value(),
                )
        return headers_

    @root_validator(pre=True)
    def validate_model(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and update model arguments, including API key and formatting"""
        values["nvidia_api_key"] = get_from_dict_or_env(
            values,
            "nvidia_api_key",
            "NVIDIA_API_KEY",
        )
        if "nvapi-" not in values.get("nvidia_api_key", ""):
            raise ValueError("Invalid NVAPI key detected. Should start with `nvapi-`")
        values["is_staging"] = "nvapi-stg-" in values["nvidia_api_key"]
        if "headers_tmpl" not in values:
            call_kvs = {
                "Accept": "application/json",
            }
            stream_kvs = {
                "Accept": "text/event-stream",
                "content-type": "application/json",
            }
            shared_kvs = {
                "Authorization": "Bearer {nvidia_api_key}",
                "User-Agent": "langchain-nvidia-ai-endpoints",
            }
            values["headers_tmpl"] = {
                "call": {**call_kvs, **shared_kvs},
                "stream": {**stream_kvs, **shared_kvs},
            }
        return values

    @root_validator(pre=False)
    def validate_model_post(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Additional validation after default values have been put in"""
        values["stagify"] = partial(cls._stagify, is_staging=values["is_staging"])
        values["fetch_url_format"] = values["stagify"](values.get("fetch_url_format"))
        values["call_invoke_base"] = values["stagify"](values.get("call_invoke_base"))
        return values

    @property
    def available_models(self) -> dict:
        """List the available models that can be invoked."""
        if self._available_models is not None:
            return self._available_models
        live_fns = [v for v in self.available_functions if v.get("status") == "ACTIVE"]
        self._available_models = {v["name"]: v["id"] for v in live_fns}
        return self._available_models

    @property
    def available_functions(self) -> List[dict]:
        """List the available functions that can be invoked."""
        if self._available_functions is not None:
            return self._available_functions
        invoke_url = self._stagify(self.func_list_format, self.is_staging)
        query_res = self.query(invoke_url)
        if "functions" not in query_res:
            raise ValueError(
                f"Unexpected response when querying {invoke_url}\n{query_res}"
            )
        self._available_functions = query_res["functions"]
        return self._available_functions

    @staticmethod
    def _stagify(path: str, is_staging: bool) -> str:
        """Helper method to switch between staging and production endpoints"""
        if is_staging and "stg.api" not in path:
            return path.replace("api.", "stg.api.")
        if not is_staging and "stg.api" in path:
            return path.replace("stg.api.", "api.")
        return path

    ####################################################################################
    ## Core utilities for posting and getting from NV Endpoints

    def _post(self, invoke_url: str, payload: dict = {}) -> Tuple[Response, Any]:
        """Method for posting to the AI Foundation Model Function API."""
        self.last_inputs = {
            "url": invoke_url,
            "headers": self.headers["call"],
            "json": self.payload_fn(payload),
            "stream": False,
        }
        session = self.get_session_fn()
        response = session.post(**self.last_inputs)
        self._try_raise(response)
        return response, session

    def _get(self, invoke_url: str, payload: dict = {}) -> Tuple[Response, Any]:
        """Method for getting from the AI Foundation Model Function API."""
        self.last_inputs = {
            "url": invoke_url,
            "headers": self.headers["call"],
            "json": self.payload_fn(payload),
            "stream": False,
        }
        session = self.get_session_fn()
        last_response = session.get(**self.last_inputs)
        self._try_raise(last_response)
        return last_response, session

    def _wait(self, response: Response, session: Any) -> Response:
        """Wait for a response from API after an initial response is made"""
        start_time = time.time()
        while response.status_code == 202:
            time.sleep(self.interval)
            if (time.time() - start_time) > self.timeout:
                raise TimeoutError(
                    f"Timeout reached without a successful response."
                    f"\nLast response: {str(response)}"
                )
            request_id = response.headers.get("NVCF-REQID", "")
            response = session.get(
                self.fetch_url_format + request_id,
                headers=self.headers["call"],
            )
        self._try_raise(response)
        return response

    def _try_raise(self, response: Response) -> None:
        """Try to raise an error from a response"""
        ## (VK) Several systems can throw errors. This tries to coerce all of them
        ## If we can't predictably pull out request id, then dump response
        try:
            response.raise_for_status()
        except requests.HTTPError:
            try:
                rd = response.json()
                if "detail" in rd and "reqId" in rd.get("detail", ""):
                    rd_buf = "- " + str(rd["detail"])
                    rd_buf = rd_buf.replace(": ", ", Error: ").replace(", ", "\n- ")
                    rd["detail"] = rd_buf
            except json.JSONDecodeError:
                rd = response.__dict__
                rd = rd.get("_content", rd)
                if isinstance(rd, bytes):
                    rd = rd.decode("utf-8")[5:]  ## remove "data:" prefix
                try:
                    rd = json.loads(rd)
                except Exception:
                    rd = {"detail": rd}
            status = rd.get("status", "###")
            title = rd.get("title", rd.get("error", "Unknown Error"))
            header = f"[{status}] {title}"
            body = ""
            if "requestId" in rd:
                if "detail" in rd:
                    body += f"{rd['detail']}\n"
                body += "RequestID: " + rd["requestId"]
            else:
                body = rd.get("detail", rd)
            if str(status) == "401":
                body += "\nPlease check or regenerate your API key."
            raise Exception(f"{header}\n{body}") from None

    ####################################################################################
    ## Simple query interface to show the set of model options

    def query(self, invoke_url: str, payload: dict = {}) -> dict:
        """Simple method for an end-to-end get query. Returns result dictionary"""
        response, session = self._get(invoke_url, payload)
        response = self._wait(response, session)
        output = self._process_response(response)[0]
        return output

    def _process_response(self, response: Union[str, Response]) -> List[dict]:
        """General-purpose response processing for single responses and streams"""
        if hasattr(response, "json"):  ## For single response (i.e. non-streaming)
            try:
                return [response.json()]
            except json.JSONDecodeError:
                response = str(response.__dict__)
        if isinstance(response, str):  ## For set of responses (i.e. streaming)
            msg_list = []
            for msg in response.split("\n\n"):
                if "{" not in msg:
                    continue
                msg_list += [json.loads(msg[msg.find("{") :])]
            return msg_list
        raise ValueError(f"Received ill-formed response: {response}")

    def _get_invoke_url(
        self, model_name: Optional[str] = None, invoke_url: Optional[str] = None
    ) -> str:
        """Helper method to get invoke URL from a model name, URL, or endpoint stub"""
        if not invoke_url:
            if not model_name:
                raise ValueError("URL or model name must be specified to invoke")
            if model_name in self.available_models:
                invoke_url = self.available_models[model_name]
            elif f"playground_{model_name}" in self.available_models:
                invoke_url = self.available_models[f"playground_{model_name}"]
            else:
                available_models_str = "\n".join(
                    [f"{k} - {v}" for k, v in self.available_models.items()]
                )
                raise ValueError(
                    f"Unknown model name {model_name} specified."
                    "\nAvailable models are:\n"
                    f"{available_models_str}"
                )
        if not invoke_url:
            # For mypy
            raise ValueError("URL or model name must be specified to invoke")
        # Why is this even needed?
        if "http" not in invoke_url:
            invoke_url = f"{self.call_invoke_base}/{invoke_url}"
        return invoke_url

    ####################################################################################
    ## Generation interface to allow users to generate new values from endpoints

    def get_req(
        self,
        model_name: Optional[str] = None,
        payload: dict = {},
        invoke_url: Optional[str] = None,
        stop: Optional[Sequence[str]] = None,
    ) -> Response:
        """Post to the API."""
        invoke_url = self._get_invoke_url(model_name, invoke_url)
        if payload.get("stream", False) is True:
            payload = {**payload, "stream": False}
        response, session = self._post(invoke_url, payload)
        return self._wait(response, session)

    def get_req_generation(
        self,
        model_name: Optional[str] = None,
        payload: dict = {},
        invoke_url: Optional[str] = None,
        stop: Optional[Sequence[str]] = None,
    ) -> dict:
        """Method for an end-to-end post query with NVE post-processing."""
        response = self.get_req(model_name, payload, invoke_url)
        output, _ = self.postprocess(response, stop=stop)
        return output

    def postprocess(
        self, response: Union[str, Response], stop: Optional[Sequence[str]] = None
    ) -> Tuple[dict, bool]:
        """Parses a response from the AI Foundation Model Function API.
        Strongly assumes that the API will return a single response.
        """
        msg_list = self._process_response(response)
        msg, is_stopped = self._aggregate_msgs(msg_list)
        msg, is_stopped = self._early_stop_msg(msg, is_stopped, stop=stop)
        return msg, is_stopped

    def _aggregate_msgs(self, msg_list: Sequence[dict]) -> Tuple[dict, bool]:
        """Dig out relevant details of aggregated message"""
        content_buffer: Dict[str, Any] = dict()
        content_holder: Dict[Any, Any] = dict()
        is_stopped = False
        for msg in msg_list:
            if "choices" in msg:
                ## Tease out ['choices'][0]...['delta'/'message']
                msg = msg.get("choices", [{}])[0]
                is_stopped = msg.get("finish_reason", "") == "stop"
                msg = msg.get("delta", msg.get("message", {"content": ""}))
            elif "data" in msg:
                ## Tease out ['data'][0]...['embedding']
                msg = msg.get("data", [{}])[0]
            content_holder = msg
            for k, v in msg.items():
                if k in ("content",) and k in content_buffer:
                    content_buffer[k] += v
                else:
                    content_buffer[k] = v
            if is_stopped:
                break
        content_holder = {**content_holder, **content_buffer}
        return content_holder, is_stopped

    def _early_stop_msg(
        self, msg: dict, is_stopped: bool, stop: Optional[Sequence[str]] = None
    ) -> Tuple[dict, bool]:
        """Try to early-terminate streaming or generation by iterating over stop list"""
        content = msg.get("content", "")
        if content and stop:
            for stop_str in stop:
                if stop_str and stop_str in content:
                    msg["content"] = content[: content.find(stop_str) + 1]
                    is_stopped = True
        return msg, is_stopped

    ####################################################################################
    ## Streaming interface to allow you to iterate through progressive generations

    def get_req_stream(
        self,
        model: Optional[str] = None,
        payload: dict = {},
        invoke_url: Optional[str] = None,
        stop: Optional[Sequence[str]] = None,
    ) -> Iterator:
        invoke_url = self._get_invoke_url(model, invoke_url)
        if payload.get("stream", True) is False:
            payload = {**payload, "stream": True}
        self.last_inputs = {
            "url": invoke_url,
            "headers": self.headers["stream"],
            "json": payload,
            "stream": True,
        }
        response = self.get_session_fn().post(**self.last_inputs)
        self._try_raise(response)
        call = self.copy()

        def out_gen() -> Generator[dict, Any, Any]:
            ## Good for client, since it allows self.last_inputs
            for line in response.iter_lines():
                if line and line.strip() != b"data: [DONE]":
                    line = line.decode("utf-8")
                    msg, final_line = call.postprocess(line, stop=stop)
                    yield msg
                    if final_line:
                        break
                self._try_raise(response)

        return (r for r in out_gen())

    ####################################################################################
    ## Asynchronous streaming interface to allow multiple generations to happen at once.

    async def get_req_astream(
        self,
        model: Optional[str] = None,
        payload: dict = {},
        invoke_url: Optional[str] = None,
        stop: Optional[Sequence[str]] = None,
    ) -> AsyncIterator:
        invoke_url = self._get_invoke_url(model, invoke_url)
        if payload.get("stream", True) is False:
            payload = {**payload, "stream": True}
        self.last_inputs = {
            "url": invoke_url,
            "headers": self.headers["stream"],
            "json": payload,
        }
        async with self.get_asession_fn() as session:
            async with session.post(**self.last_inputs) as response:
                self._try_raise(response)
                async for line in response.content.iter_any():
                    if line and line.strip() != b"data: [DONE]":
                        line = line.decode("utf-8")
                        msg, final_line = self.postprocess(line, stop=stop)
                        yield msg
                        if final_line:
                            break


class _NVIDIAClient(BaseModel):
    """
    Higher-Level AI Foundation Model Function API Client with argument defaults.
    Is subclassed by ChatNVIDIA to provide a simple LangChain interface.
    """

    client: NVEModel = Field(NVEModel)

    model: str = Field(..., description="Name of the model to invoke")

    ####################################################################################

    @root_validator(pre=True)
    def validate_client(cls, values: Any) -> Any:
        """Validate and update client arguments, including API key and formatting"""
        if not values.get("client"):
            values["client"] = NVEModel(**values)
        return values

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @property
    def available_functions(self) -> List[dict]:
        """Map the available functions that can be invoked."""
        return self.client.available_functions

    @property
    def available_models(self) -> dict:
        """Map the available models that can be invoked."""
        return self.client.available_models

    @staticmethod
    def get_available_functions(**kwargs: Any) -> List[dict]:
        """Map the available functions that can be invoked. Callable from class"""
        return NVEModel(**kwargs).available_functions

    @staticmethod
    def get_available_models(**kwargs: Any) -> dict:
        """Map the available models that can be invoked. Callable from class"""
        return NVEModel(**kwargs).available_models

    def get_model_details(self, model: Optional[str] = None) -> dict:
        """Get more meta-details about a model retrieved by a given name"""
        if model is None:
            model = self.model
        model_key = self.client._get_invoke_url(model).split("/")[-1]
        known_fns = self.client.available_functions
        fn_spec = [f for f in known_fns if f.get("id") == model_key][0]
        return fn_spec
