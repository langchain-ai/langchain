from __future__ import annotations

import base64
import io
import logging
import os
import sys
import urllib.parse
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Type,
    Union,
)

import requests
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import LLM
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    ChatMessageChunk,
)
from langchain_core.outputs import (
    Generation,
    GenerationChunk,
    LLMResult,
)
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import Runnable
from langchain_core.runnables.config import run_in_executor
from langchain_core.tools import BaseTool

from langchain_nvidia_ai_endpoints.nvcf import _common as nvidia_ai_endpoints

_CallbackManager = Union[AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun]
_DictOrPydanticClass = Union[Dict[str, Any], Type[BaseModel]]
_DictOrPydantic = Union[Dict, BaseModel]

try:
    import PIL.Image

    has_pillow = True
except ImportError:
    has_pillow = False


class NVCF(nvidia_ai_endpoints._NVIDIAClient, LLM):
    """NVIDIA chat model.

    Example:
        .. code-block:: python

            from langchain_nvidia_ai_endpoints import ChatNVIDIA


            model = ChatNVIDIA(model="llama2_13b")
            response = model.invoke("Hello")
    """

    temperature: Optional[float] = Field(description="Sampling temperature in [0, 1]")
    max_tokens: Optional[int] = Field(description="Maximum # of tokens to generate")
    top_p: Optional[float] = Field(description="Top-p for distribution sampling")
    frequency_penalty: Optional[float] = Field(description="Frequency penalty")
    presence_penalty: Optional[float] = Field(description="Presence penalty")
    seed: Optional[int] = Field(description="The seed for deterministic results")
    bad: Optional[Sequence[str]] = Field(description="Bad words to avoid (cased)")
    stop: Optional[Sequence[str]] = Field(description="Stop words (cased)")
    labels: Optional[Dict[str, float]] = Field(description="Steering parameters")
    streaming: bool = Field(True)

    @property
    def _llm_type(self) -> str:
        """Return type of NVIDIA AI Foundation Model Interface."""
        return "nvidia-ai-playground"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Invoke on a single list of chat messages."""
        response = self.get_generation(prompt=prompt, stop=stop, **kwargs)
        output = self.custom_postprocess(response)
        return output

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """Allows streaming to model!"""
        for response in self.get_stream(prompt=prompt, stop=stop, **kwargs):
            self._set_callback_out(response, run_manager)
            chunk = GenerationChunk(text=self.custom_postprocess(response))
            yield chunk
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)

    async def _astream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[GenerationChunk]:
        async for response in self.get_astream(prompt=prompt, stop=stop, **kwargs):
            self._set_callback_out(response, run_manager)
            chunk = GenerationChunk(text=self.custom_postprocess(response))
            yield chunk
            if run_manager:
                await run_manager.on_llm_new_token(chunk.text, chunk=chunk)

    def _set_callback_out(
        self,
        result: dict,
        run_manager: Optional[_CallbackManager],
    ) -> None:
        result.update({"model_name": self.model})
        if run_manager:
            for cb in run_manager.handlers:
                if hasattr(cb, "llm_output"):
                    cb.llm_output = result

    def custom_postprocess(self, msg: dict) -> str:
        if "content" in msg:
            return msg["content"]
        elif "b64_json" in msg:
            return msg["b64_json"]
        return str(msg)

    ######################################################################################
    ## Core client-side interfaces

    def get_generation(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> dict:
        """Call to client generate method with call scope"""
        stop = kwargs.get("stop", None)
        payload = self.get_payload(prompt=prompt, stream=False, **kwargs)
        out = self.client.get_req_generation(self.model, stop=stop, payload=payload)
        return out

    def get_stream(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> Iterator:
        """Call to client stream method with call scope"""
        stop = kwargs.get("stop", None)
        payload = self.get_payload(prompt=prompt, stream=True, **kwargs)
        return self.client.get_req_stream(self.model, stop=stop, payload=payload)

    def get_astream(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> AsyncIterator:
        """Call to client astream methods with call scope"""
        stop = kwargs.get("stop", None)
        payload = self.get_payload(prompt=prompt, stream=True, **kwargs)
        return self.client.get_req_astream(self.model, stop=stop, payload=payload)

    def get_payload(self, prompt: str, **kwargs: Any) -> dict:
        """Generates payload for the _NVIDIAClient API to send to service."""
        attr_kwargs = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "seed": self.seed,
            "bad": self.bad,
            "stop": self.stop,
            "labels": self.labels,
        }
        attr_kwargs = {k: v for k, v in attr_kwargs.items() if v is not None}
        new_kwargs = {**attr_kwargs, **kwargs}
        return self.prep_payload(prompt=prompt, **new_kwargs)

    def prep_payload(self, prompt: str, **kwargs: Any) -> dict:
        """Prepares a message or list of messages for the payload"""
        if kwargs.get("stop") is None:
            kwargs.pop("stop")
        return {"prompt": prompt, **kwargs}
