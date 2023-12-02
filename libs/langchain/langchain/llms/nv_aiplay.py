## NOTE: This class is intentionally implemented to subclass either ChatModel or LLM for
##  demonstrative purposes and to make it function as a simple standalone file.

from __future__ import annotations

import asyncio
import json
import logging
import re
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
from requests.models import Response

from langchain.callbacks.manager import (
    AsyncCallbackManager,
    AsyncCallbackManagerForLLMRun,
    CallbackManager,
)
from langchain.llms.base import LLM
from langchain.pydantic_v1 import BaseModel, Field, SecretStr, root_validator
from langchain.schema.messages import BaseMessage, ChatMessageChunk
from langchain.schema.output import ChatGenerationChunk, GenerationChunk
from langchain.utils import get_from_dict_or_env

logger = logging.getLogger(__name__)


class ClientModel(BaseModel):
    """
    Custom BaseModel subclass with some desirable properties for subclassing
    """

    saved_parent: Optional[ClientModel] = None

    def __init__(self, *args: Sequence, **kwargs: Any[str, Any]):
        super().__init__(*args, **kwargs)

    def subscope(self, *args: Sequence, **kwargs: Any) -> Any:
        """Create a new ClientModel with the same values but new arguments"""
        named_args = dict({k: v for k, v in zip(getattr(self, "arg_keys", []), args)})
        named_args = {**named_args, **kwargs}
        out = self.copy(update=named_args)
        out.validate(dict(out._iter(to_dict=False, by_alias=False, exclude_unset=True)))
        for k, v in self.__dict__.items():
            if isinstance(v, ClientModel):
                setattr(out, k, v.subscope(*args, **kwargs))
        out.saved_parent = self
        return out

    def dict(self, *args: Sequence, **kwargs: Any) -> dict:
        """Handle saved_parent bleeding into dict"""
        out = super().dict(*args, **kwargs)
        if "saved_parent" in out:
            out.pop("saved_parent")
        return out

    def get(self, key: str) -> Any:
        """Get a value from the ClientModel, using it like a dictionary"""
        return getattr(self, key)

    def transfer_state(self, other: Optional[ClientModel]) -> None:
        """Transfer state from one ClientModel to another"""
        if other is None:
            return
        for k, v in self.__dict__.items():
            if k in getattr(self, "state_vars", []):
                setattr(other, k, v)
            elif hasattr(v, "transfer_state"):
                other_sub = getattr(other, k, None)
                if other_sub is not None:
                    v.transfer_state(other_sub)

    @staticmethod
    def desecretize(v: Any) -> Any:
        """Desecretize a collection of values"""
        recurse = ClientModel.desecretize
        if isinstance(v, SecretStr):
            return v.get_secret_value()
        if isinstance(v, str):
            return v
        if isinstance(v, dict):
            return {k: recurse(v) for k, v in v.items()}
        if isinstance(v, list):
            return [recurse(subv) for subv in v]
        if isinstance(v, tuple):
            return tuple(recurse(subv) for subv in v)
        return v

    def __enter__(self) -> ClientModel:
        return self

    def __exit__(self, type: Any, value: Any, traceback: Any) -> None:
        self.transfer_state(self.saved_parent)
        self.saved_parent = None


class NVCRModel(ClientModel):

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

    ## Populated on construction/validation
    nvapi_key: Optional[SecretStr]
    is_staging: Optional[bool]
    available_models: Optional[Dict[str, str]]

    ## Generation arguments
    max_tries: int = Field(5, ge=1)
    stop: Union[str, List[str]] = Field([])
    headers = dict(
        call={"Authorization": "Bearer {nvapi_key}", "Accept": "application/json"},
        stream={
            "Authorization": "Bearer {nvapi_key}",
            "Accept": "text/event-stream",
            "content-type": "application/json",
        },
    )

    ## Status Tracking Variables. Updated Progressively
    last_inputs: Optional[dict] = Field(None)
    last_response: Optional[Any] = Field(None)
    last_msg: dict = Field({})
    available_functions: List[dict] = Field([{}])
    state_vars: Sequence[str] = Field(
        [
            "last_inputs",
            "last_response",
            "last_msg",
            "available_functions",
        ]
    )

    @root_validator()
    def validate_model(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and update model arguments, including API key and formatting"""
        values["nvapi_key"] = get_from_dict_or_env(values, "nvapi_key", "NVAPI_KEY")
        if "nvapi-" not in values.get("nvapi_key", ""):
            raise ValueError("Invalid NVAPI key detected. Should start with `nvapi-`")
        values["is_staging"] = "nvapi-stg-" in values["nvapi_key"]
        for header in values["headers"].values():
            if "{nvapi_key}" in header["Authorization"]:
                nvapi_key = ClientModel.desecretize(values["nvapi_key"])
                header["Authorization"] = SecretStr(
                    header["Authorization"].format(nvapi_key=nvapi_key),
                )
        if isinstance(values["stop"], str):
            values["stop"] = [values["stop"]]
        return values

    def __init__(self, *args: Sequence, **kwargs: Any):
        """Useful to define custom operations on construction after validation"""
        super().__init__(*args, **kwargs)
        self.fetch_url_format = self._stagify(self.fetch_url_format)
        self.call_invoke_base = self._stagify(self.call_invoke_base)
        try:
            self.available_models = self.get_available_models()
        except Exception as e:
            raise Exception("Error retrieving model list. Verify your NVAPI key") from e

    def _stagify(self, path: str) -> str:
        """Helper method to switch between staging and production endpoints"""
        if self.is_staging and "stg.api" not in path:
            return path.replace("api", "stg.api")
        if not self.is_staging and "stg.api" in path:
            return path.replace("stg.api", "api")
        return path

    ####################################################################################
    ## Core utilities for posting and getting from NVCR

    def _post(self, invoke_url: str, payload: dict = {}) -> Tuple[Response, Any]:
        """Method for posting to the AI Playground API."""
        self.last_inputs = dict(
            url=invoke_url,
            headers=self.headers["call"],
            json=payload,
            stream=False,
        )
        session = self.get_session_fn()
        self.last_response = session.post(**ClientModel.desecretize(self.last_inputs))
        self._try_raise(self.last_response)
        return self.last_response, session

    def _get(self, invoke_url: str, payload: dict = {}) -> Tuple[Response, Any]:
        """Method for getting from the AI Playground API."""
        self.last_inputs = dict(
            url=invoke_url,
            headers=self.headers["call"],
            json=payload,
            stream=False,
        )
        session = self.get_session_fn()
        self.last_response = session.get(**ClientModel.desecretize(self.last_inputs))
        self._try_raise(self.last_response)
        return self.last_response, session

    def _wait(self, response: Response, session: Any) -> Response:
        """Wait for a response from API after an initial response is made."""
        i = 1
        while response.status_code == 202:
            request_id = response.headers.get("NVCF-REQID", "")
            response = session.get(
                self.fetch_url_format + request_id,
                headers=ClientModel.desecretize(self.headers["call"]),
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
                    rd = rd.decode("utf-8")[5:]  ## lop of data: prefix
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

    def get_available_models(self) -> dict:
        """Get a dictionary of available models from the AI Playground API."""
        invoke_url = self._stagify("https://api.nvcf.nvidia.com/v2/nvcf/functions")
        self.available_functions = self.query(invoke_url)["functions"]
        live_fns = [v for v in self.available_functions if v.get("status") == "ACTIVE"]
        return {v["name"]: v["id"] for v in live_fns}

    def _get_invoke_url(
        self, model_name: Optional[str] = None, invoke_url: Optional[str] = None
    ) -> str:
        """Helper method to get invoke URL from a model name, URL, or endpoint stub"""
        if not invoke_url:
            if not model_name:
                raise ValueError("URL or model name must be specified to invoke")
            available_models = self.available_models or self.get_available_models()
            if model_name in available_models:
                invoke_url = available_models.get(model_name)
            else:
                for key in sorted(available_models.keys()):
                    if model_name in key:
                        invoke_url = available_models[key]
                        break
        if not invoke_url:
            raise ValueError(f"Unknown model name {model_name} specified")
        if "http" not in invoke_url:
            invoke_url = f"{self.call_invoke_base}/{invoke_url}"
        return invoke_url

    ####################################################################################
    ## Generation interface to allow users to generate new values from endpoints

    def get_req_generation(
        self,
        model_name: Optional[str] = None,
        payload: dict = {},
        invoke_url: Optional[str] = None,
    ) -> dict:
        """Method for an end-to-end post query with NVCR post-processing."""
        invoke_url = self._get_invoke_url(model_name, invoke_url)
        if payload.get("stream", False) is True:
            payload = {**payload, "stream": False}
        response, session = self._post(invoke_url, payload)
        response = self._wait(response, session)
        output, _ = self.postprocess(response)
        return output

    def postprocess(self, response: Union[str, Response]) -> Tuple[dict, bool]:
        """Parses a response from the AI Playground API.
        Strongly assumes that the API will return a single response.
        """
        msg_list = self._process_response(response)
        msg, is_stopped = self._aggregate_msgs(msg_list)
        msg, is_stopped = self._early_stop_msg(msg, is_stopped)
        return msg, is_stopped

    def _aggregate_msgs(self, msg_list: Sequence[dict]) -> Tuple[dict, bool]:
        """Dig out relevant details of aggregated message"""
        content_buffer: Dict[str, Any] = dict()
        content_holder: Dict[Any, Any] = dict()
        is_stopped = False
        for msg in msg_list:
            self.last_msg = msg
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

    def _early_stop_msg(self, msg: dict, is_stopped: bool) -> Tuple[dict, bool]:
        """Try to early-terminate streaming or generation by iterating over stop list"""
        content = msg.get("content", "")
        if content and self.stop:
            for stop_str in self.stop:
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
    ) -> Iterator:
        invoke_url = self._get_invoke_url(model, invoke_url)
        if payload.get("stream", True) is False:
            payload = {**payload, "stream": True}
        self.last_inputs = dict(
            url=invoke_url,
            headers=self.headers["stream"],
            json=payload,
            stream=True,
        )
        raw_inputs = ClientModel.desecretize(self.last_inputs)
        response = self.get_session_fn().post(**raw_inputs)
        self.last_response = response
        self._try_raise(response)
        call = self.copy()

        def out_gen() -> Generator[dict, Any, Any]:
            ## Good for client, since it allows self.last_input
            for line in response.iter_lines():
                if line and line.strip() != b"data: [DONE]":
                    line = line.decode("utf-8")
                    msg, final_line = call.postprocess(line)
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
    ) -> AsyncIterator:
        invoke_url = self._get_invoke_url(model, invoke_url)
        if payload.get("stream", True) is False:
            payload = {**payload, "stream": True}
        self.last_inputs = dict(
            url=invoke_url,
            headers=self.headers["stream"],
            json=payload,
        )
        async with self.get_asession_fn() as session:
            raw_inputs = ClientModel.desecretize(self.last_inputs)
            async with session.post(**raw_inputs) as self.last_response:
                self._try_raise(self.last_response)
                async for line in self.last_response.content.iter_any():
                    if line and line.strip() != b"data: [DONE]":
                        line = line.decode("utf-8")
                        msg, final_line = self.postprocess(line)
                        yield msg
                        if final_line:
                            break


class NVAIPlayClient(ClientModel):
    """
    Higher-Level Client for interacting with AI Playground API with argument defaults.
    Is subclassed by NVAIPlayLLM/NVAIPlayChat to provide a simple LangChain interface.
    """

    client: NVCRModel = Field(NVCRModel)

    model: str = Field("llama")
    labels: dict = Field({})

    temperature: float = Field(0.2, le=1.0, gt=0.0)
    top_p: float = Field(0.7, le=1.0, ge=0.0)
    max_tokens: int = Field(1024, le=1024, ge=32)
    streaming: bool = Field(False)

    inputs: Any = Field([])
    stop: Union[Sequence[str], str] = Field([])

    gen_keys: Sequence[str] = Field(["temperature", "top_p", "max_tokens", "streaming"])
    arg_keys: Sequence[str] = Field(["inputs", "stop"])
    valid_roles: Sequence[str] = Field(["user", "system", "assistant"])

    class LabelModel(ClientModel):
        creativity: int = Field(0, ge=0, le=9)
        complexity: int = Field(0, ge=0, le=9)
        verbosity: int = Field(0, ge=0, le=9)

    ####################################################################################

    def __init__(self, *args: Sequence, **kwargs: Any):
        super().__init__(*args, **kwargs)

    @root_validator()
    def validate_model(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        values["client"] = values["client"](**values)
        if values.get("labels"):
            values["labels"] = cls.LabelModel(**values["labels"]).dict()
        return values

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @property
    def available_models(self) -> List[str]:
        """List the available models that can be invoked"""
        return list(getattr(self.client, "available_models", {}).keys())

    def get_model_details(self, model: Optional[str] = None) -> dict:
        """Get more meta-details about a model retrieved by a given name"""
        if model is None:
            model = self.model
        model_key = self.client._get_invoke_url(model).split("/")[-1]
        known_fns = self.client.available_functions
        fn_spec = [f for f in known_fns if f.get("id") == model_key][0]
        return fn_spec

    def get_generation(self, *args: Sequence, **kwargs: Any) -> dict:
        """Call to client generate method with call scope"""
        with self.subscope(*args, **kwargs) as call:
            payload = call.get_payload(stream=False)
            out = call.client.get_req_generation(call.model, payload=payload)
        return out

    def get_stream(self, *args: Sequence, **kwargs: Any) -> Iterator:
        """Call to client stream method with call scope"""
        with self.subscope(*args, **kwargs) as call:
            payload = call.get_payload(stream=True)
            out = call.client.get_req_stream(call.model, payload=payload)
        return out

    def get_astream(self, *args: Sequence, **kwargs: Any) -> AsyncIterator:
        """Call to client astream method with call scope"""
        with self.subscope(*args, **kwargs) as call:
            payload = call.get_payload(stream=True)
            out = call.client.get_req_astream(call.model, payload=payload)
        return out

    def get_payload(self, *args: Sequence, **kwargs: Any) -> dict:
        """Generates payload for the NVAIPlayClient API to send to service."""

        def k_map(k: str) -> str:
            return k if k != "streaming" else "stream"

        out = {**self.preprocess(), **{k_map(k): self.get(k) for k in self.gen_keys}}
        return out

    def preprocess(self) -> dict:
        """Prepares a message or list of messages for the payload"""
        if (
            isinstance(self.inputs, str)
            or not hasattr(self.inputs, "__iter__")
            or isinstance(self.inputs, BaseMessage)
        ):
            self.inputs = [self.inputs]
        messages = [self.prep_msg(m) for m in self.inputs]
        labels = self.labels
        if labels:
            messages += [{"labels": labels, "role": "assistant"}]
        return {"messages": messages}

    def prep_msg(self, msg: Union[str, dict, BaseMessage]) -> dict:
        """Helper Method: Ensures a message is a dictionary with a role and content."""
        if isinstance(msg, str):
            return dict(role="user", content=msg)
        if isinstance(msg, dict):
            if msg.get("role", "") not in self.valid_roles:
                raise ValueError(f"Unknown message role \"{msg.get('role', '')}\"")
            if msg.get("content", None) is None:
                raise ValueError(f"Message {msg} has no content")
            return msg
        raise ValueError(f"Unknown message received: {msg} of type {type(msg)}")


class NVAIPlayBaseModel(NVAIPlayClient):
    """
    Base class for NVIDIA AI Playground models which can interface with NVAIPlayClient.
    To be subclassed by NVAIPlayLLM/NVAIPlayChat by combining with LLM/SimpleChatModel.
    """

    @property
    def _llm_type(self) -> str:
        """Return type of NVIDIA AI Playground Interface."""
        return "nvidia_ai_playground"

    def _call(
        self,
        messages: Union[List[BaseMessage], str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ) -> str:
        """hook for LLM/SimpleChatModel. Allows for easy standard/streaming calls"""
        kwargs["labels"] = kwargs.get("labels", self.labels)
        kwargs["stop"] = stop if stop else getattr(self.client, "stop")
        if kwargs.get("streaming", self.streaming) or kwargs["stop"]:
            buffer = ""
            for chunk in self._stream(messages, run_manager=run_manager, **kwargs):
                buffer += chunk if isinstance(chunk, str) else chunk.text
            responses = {"content": buffer}
        else:
            inputs = self.custom_preprocess(messages)
            responses = self.get_generation(inputs, **kwargs)
        outputs = self.custom_postprocess(responses)
        return outputs

    def _get_filled_chunk(
        self, text: str, role: Optional[str] = "assistant"
    ) -> Union[GenerationChunk, ChatGenerationChunk]:
        """LLM and BasicChatModel have different streaming chunk specifications"""
        if isinstance(self, LLM):
            return GenerationChunk(text=text)
        return ChatGenerationChunk(message=ChatMessageChunk(content=text, role=role))

    def _stream(
        self,
        messages: Union[List[BaseMessage], str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ) -> Iterator[Union[GenerationChunk, ChatGenerationChunk]]:
        """Allows streaming to model!"""
        inputs = self.custom_preprocess(messages)
        kwargs["labels"] = kwargs.get("labels", self.labels)
        kwargs["stop"] = stop if stop else getattr(self.client, "stop")
        for response in self.get_stream(inputs, **kwargs):
            chunk = self._get_filled_chunk(self.custom_postprocess(response))
            yield chunk
            if run_manager:
                async_mtypes = (AsyncCallbackManager, AsyncCallbackManagerForLLMRun)
                if isinstance(run_manager, async_mtypes):
                    ## Edge case from LLM/SimpleChatModel default async methods
                    asyncio.run(run_manager.on_llm_new_token(chunk.text, chunk=chunk))
                else:
                    run_manager.on_llm_new_token(chunk.text, chunk=chunk)

    async def _astream(
        self,
        messages: Union[List[BaseMessage], str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManager] = None,
        **kwargs: Any,
    ) -> AsyncIterator[Union[GenerationChunk, ChatGenerationChunk]]:
        inputs = self.custom_preprocess(messages)
        kwargs["labels"] = kwargs.get("labels", self.labels)
        kwargs["stop"] = stop if stop else getattr(self.client, "stop")
        async for response in self.get_astream(inputs, **kwargs):
            chunk = self._get_filled_chunk(self.custom_postprocess(response))
            yield chunk
            if run_manager:
                await run_manager.on_llm_new_token(chunk.text, chunk=chunk)

    def custom_preprocess(self, msgs: Union[str, Sequence]) -> List[Dict[str, str]]:
        is_one = isinstance(msgs, (str, BaseMessage))
        is_list = not is_one and hasattr(msgs, "__iter__")
        is_solo = is_list and len(msgs) == 1 and isinstance(msgs[0], (str, BaseMessage))
        msg_list: Sequence[Any] = []
        if is_one or is_solo:
            msg_val: Union[str, BaseMessage] = msgs if not is_list else msgs[0]
            msg_str: str = getattr(msg_val, "content", msg_val)
            msg_list = re.split("///ROLE ", msg_str.strip())
            msg_list = [m for m in msg_list if m.strip()]
        elif not is_list:
            msg_list = [msgs]
        elif is_list:
            msg_list = msgs
        out = [self.preprocess_msg(m) for m in msg_list]
        return out

    def preprocess_msg(
        self, msg: Union[str, Sequence[str], dict, BaseMessage]
    ) -> Dict[str, str]:
        ## Support for just simple string inputs of ///ROLE SYS etc. inputs
        if isinstance(msg, str):
            msg_split = re.split("SYS: |USER: |AGENT: |CONTEXT:", msg)
            if len(msg_split) == 1:
                return {"role": "user", "content": msg}
            role_convert = {
                "agent": "assistant",
                "sys": "system",
                "context": "context",
            }
            role, _, content = msg.partition(": ")
            role = role_convert.get(role.strip().lower(), "user")
            return {"role": role, "content": content}
        ## Support for tuple inputs
        if type(msg) in (list, tuple):
            return {"role": msg[0], "content": msg[1]}
        ## Support for manually-specified default inputs to AI Playground
        if isinstance(msg, dict) and msg.get("content"):
            msg["role"] = msg.get("role", "user")
            return msg
        ## Support for LangChain Messages
        if hasattr(msg, "content"):
            role_convert = {"ai": "assistant", "system": "system"}
            role = getattr(msg, "type")
            cont = getattr(msg, "content")
            role = role_convert.get(role, "user")
            if hasattr(msg, "role"):
                cont = f"{getattr(msg, 'role')}: {cont}"
            return {"role": role, "content": cont}
        raise ValueError(f"Invalid message: {repr(msg)} of type {type(msg)}")

    def custom_postprocess(self, msg: dict) -> str:
        if "content" in msg:
            return msg["content"]
        logger.warning(
            f"Got ambiguous message in postprocessing; returning as-is: msg = {msg}"
        )
        return str(msg)


####################################################################################


class GeneralBase(NVAIPlayBaseModel):
    model: str = Field("llama2_13b")


class CodeBase(NVAIPlayBaseModel):
    model: str = Field("llama2_code_13b")


class InstructBase(NVAIPlayBaseModel):
    model: str = Field("mistral")


class SteerBase(NVAIPlayBaseModel):
    model: str = Field("steerlm")
    arg_keys: Sequence[str] = Field(["inputs", "labels", "stop"])
    labels: dict = Field({"creativity": 0, "complexity": 9, "verbosity": 9})


class ContextBase(NVAIPlayBaseModel):
    model: str = Field("_qa_")
    valid_roles: Sequence[str] = Field(["user", "context"])
    max_tokens: int = Field(512, ge=32, le=512)


class ImageBase(NVAIPlayBaseModel):
    model: str = Field("neva")
    arg_keys: Sequence[str] = Field(["inputs", "labels", "stop"])
    labels: dict = Field({"creativity": 0, "complexity": 9, "verbosity": 9})


####################################################################################


class NVAIPlayLLM(NVAIPlayBaseModel, LLM):
    pass


class GeneralLLM(GeneralBase, LLM):
    pass


class CodeLLM(CodeBase, LLM):
    pass


class InstructLLM(InstructBase, LLM):
    pass


class SteerLLM(SteerBase, LLM):
    pass


class ContextLLM(ContextBase, LLM):
    pass


class ImageLLM(ImageBase, LLM):
    pass
