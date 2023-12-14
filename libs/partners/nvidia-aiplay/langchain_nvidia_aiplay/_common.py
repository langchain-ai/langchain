from __future__ import annotations

import json
import logging
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
from langchain_core.messages import BaseMessage
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


class NVCRModel(BaseModel):

    """
    Underlying Client for interacting with the AI Playground API.
    Leveraged by the NVAIPlayBaseModel to provide a simple requests-oriented interface.
    Direct abstraction over NGC-recommended streaming/non-streaming Python solutions.

    NOTE: AI Playground does not currently support raw text continuation.
    """

    ## Core defaults. These probably should not be changed
    fetch_url_format: str = Field("https://api.nvcf.nvidia.com/v2/nvcf/pexec/status/")
    call_invoke_base: str = Field("https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions")
    get_session_fn: Callable = Field(requests.Session)
    get_asession_fn: Callable = Field(aiohttp.ClientSession)

    nvidia_api_key: SecretStr = Field(
        ...,
        description="API key for NVIDIA AI Playground. Should start with `nvapi-`",
    )
    is_staging: bool = Field(False, description="Whether to use staging API")

    ## Generation arguments
    max_tries: int = Field(5, ge=1)
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
        is_staging = "nvapi-stg-" in values["nvidia_api_key"]
        values["is_staging"] = is_staging
        if "headers_tmpl" not in values:
            values["headers_tmpl"] = {
                "call": {
                    "Authorization": "Bearer {nvidia_api_key}",
                    "Accept": "application/json",
                },
                "stream": {
                    "Authorization": "Bearer {nvidia_api_key}",
                    "Accept": "text/event-stream",
                    "content-type": "application/json",
                },
            }

        values["fetch_url_format"] = cls._stagify(
            is_staging,
            values.get(
                "fetch_url_format", "https://api.nvcf.nvidia.com/v2/nvcf/pexec/status/"
            ),
        )
        values["call_invoke_base"] = cls._stagify(
            is_staging,
            values.get(
                "call_invoke_base",
                "https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions",
            ),
        )
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
        invoke_url = self._stagify(
            self.is_staging, "https://api.nvcf.nvidia.com/v2/nvcf/functions"
        )
        query_res = self.query(invoke_url)
        if "functions" not in query_res:
            raise ValueError(
                f"Unexpected response when querying {invoke_url}\n{query_res}"
            )
        self._available_functions = query_res["functions"]
        return self._available_functions

    @classmethod
    def _stagify(cls, is_staging: bool, path: str) -> str:
        """Helper method to switch between staging and production endpoints"""
        if is_staging and "stg.api" not in path:
            return path.replace("api.", "stg.api.")
        if not is_staging and "stg.api" in path:
            return path.replace("stg.api.", "api.")
        return path

    ####################################################################################
    ## Core utilities for posting and getting from NVCR

    def _post(self, invoke_url: str, payload: dict = {}) -> Tuple[Response, Any]:
        """Method for posting to the AI Playground API."""
        call_inputs = {
            "url": invoke_url,
            "headers": self.headers["call"],
            "json": payload,
            "stream": False,
        }
        session = self.get_session_fn()
        response = session.post(**call_inputs)
        self._try_raise(response)
        return response, session

    def _get(self, invoke_url: str, payload: dict = {}) -> Tuple[Response, Any]:
        """Method for getting from the AI Playground API."""
        last_inputs = {
            "url": invoke_url,
            "headers": self.headers["call"],
            "json": payload,
            "stream": False,
        }
        session = self.get_session_fn()
        last_response = session.get(**last_inputs)
        self._try_raise(last_response)
        return last_response, session

    def _wait(self, response: Response, session: Any) -> Response:
        """Wait for a response from API after an initial response is made."""
        i = 1
        while response.status_code == 202:
            request_id = response.headers.get("NVCF-REQID", "")
            response = session.get(
                self.fetch_url_format + request_id,
                headers=self.headers["call"],
            )
            if response.status_code == 202:
                try:
                    body = response.json()
                except ValueError:
                    body = str(response)
                if i > self.max_tries:
                    raise ValueError(f"Failed to get response with {i} tries: {body}")
        self._try_raise(response)
        return response

    def _try_raise(self, response: Response) -> None:
        """Try to raise an error from a response"""
        try:
            response.raise_for_status()
        except requests.HTTPError as e:
            try:
                rd = response.json()
            except json.JSONDecodeError:
                rd = response.__dict__
                rd = rd.get("_content", rd)
                if isinstance(rd, bytes):
                    rd = rd.decode("utf-8")[5:]  ## lop of data: prefix ??
                try:
                    rd = json.loads(rd)
                except Exception:
                    rd = {"detail": rd}
            title = f"[{rd.get('status', '###')}] {rd.get('title', 'Unknown Error')}"
            body = f"{rd.get('detail', rd.get('type', rd))}"
            raise Exception(f"{title}\n{body}") from e

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
        """Method for an end-to-end post query with NVCR post-processing."""
        response = self.get_req(model_name, payload, invoke_url)
        output, _ = self.postprocess(response, stop=stop)
        return output

    def postprocess(
        self, response: Union[str, Response], stop: Optional[Sequence[str]] = None
    ) -> Tuple[dict, bool]:
        """Parses a response from the AI Playground API.
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
        last_inputs = {
            "url": invoke_url,
            "headers": self.headers["stream"],
            "json": payload,
            "stream": True,
        }
        response = self.get_session_fn().post(**last_inputs)
        self._try_raise(response)
        call = self.copy()

        def out_gen() -> Generator[dict, Any, Any]:
            ## Good for client, since it allows self.last_input
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
        last_inputs = {
            "url": invoke_url,
            "headers": self.headers["stream"],
            "json": payload,
        }
        async with self.get_asession_fn() as session:
            async with session.post(**last_inputs) as response:
                self._try_raise(response)
                async for line in response.content.iter_any():
                    if line and line.strip() != b"data: [DONE]":
                        line = line.decode("utf-8")
                        msg, final_line = self.postprocess(line, stop=stop)
                        yield msg
                        if final_line:
                            break


class _NVAIPlayClient(BaseModel):
    """
    Higher-Level Client for interacting with AI Playground API with argument defaults.
    Is subclassed by NVAIPlayLLM/ChatNVAIPlay to provide a simple LangChain interface.
    """

    client: NVCRModel = Field(NVCRModel)

    model: str = Field(..., description="Name of the model to invoke")

    temperature: float = Field(0.2, le=1.0, gt=0.0)
    top_p: float = Field(0.7, le=1.0, ge=0.0)
    max_tokens: int = Field(1024, le=1024, ge=32)

    ####################################################################################

    @root_validator(pre=True)
    def validate_client(cls, values: Any) -> Any:
        """Validate and update client arguments, including API key and formatting"""
        if not values.get("client"):
            values["client"] = NVCRModel(**values)
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

    def get_model_details(self, model: Optional[str] = None) -> dict:
        """Get more meta-details about a model retrieved by a given name"""
        if model is None:
            model = self.model
        model_key = self.client._get_invoke_url(model).split("/")[-1]
        known_fns = self.client.available_functions
        fn_spec = [f for f in known_fns if f.get("id") == model_key][0]
        return fn_spec

    def get_generation(
        self,
        inputs: Sequence[Dict],
        labels: Optional[dict] = None,
        stop: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> dict:
        """Call to client generate method with call scope"""
        payload = self.get_payload(inputs=inputs, stream=False, labels=labels, **kwargs)
        out = self.client.get_req_generation(self.model, stop=stop, payload=payload)
        return out

    def get_stream(
        self,
        inputs: Sequence[Dict],
        labels: Optional[dict] = None,
        stop: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> Iterator:
        """Call to client stream method with call scope"""
        payload = self.get_payload(inputs=inputs, stream=True, labels=labels, **kwargs)
        return self.client.get_req_stream(self.model, stop=stop, payload=payload)

    def get_astream(
        self,
        inputs: Sequence[Dict],
        labels: Optional[dict] = None,
        stop: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> AsyncIterator:
        """Call to client astream methods with call scope"""
        payload = self.get_payload(inputs=inputs, stream=True, labels=labels, **kwargs)
        return self.client.get_req_astream(self.model, stop=stop, payload=payload)

    def get_payload(
        self, inputs: Sequence[Dict], labels: Optional[dict] = None, **kwargs: Any
    ) -> dict:
        """Generates payload for the _NVAIPlayClient API to send to service."""
        return {
            **self.preprocess(inputs=inputs, labels=labels),
            **kwargs,
        }

    def preprocess(self, inputs: Sequence[Dict], labels: Optional[dict] = None) -> dict:
        """Prepares a message or list of messages for the payload"""
        messages = [self.prep_msg(m) for m in inputs]
        if labels:
            # (WFH) Labels are currently (?) always passed as an assistant
            # suffix message, but this API seems less stable.
            messages += [{"labels": labels, "role": "assistant"}]
        return {"messages": messages}

    def prep_msg(self, msg: Union[str, dict, BaseMessage]) -> dict:
        """Helper Method: Ensures a message is a dictionary with a role and content."""
        if isinstance(msg, str):
            # (WFH) this shouldn't ever be reached but leaving this here bcs
            # it's a Chesterton's fence I'm unwilling to touch
            return dict(role="user", content=msg)
        if isinstance(msg, dict):
            if msg.get("content", None) is None:
                raise ValueError(f"Message {msg} has no content")
            return msg
        raise ValueError(f"Unknown message received: {msg} of type {type(msg)}")
