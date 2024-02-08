from __future__ import annotations

import base64
import logging
import os
from io import BytesIO
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    cast,
)
from urllib.parse import urlparse

import google.api_core

# TODO: remove ignore once the google package is published with types
import google.generativeai as genai  # type: ignore[import]
import requests
from google.ai.generativelanguage_v1beta import FunctionCall
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ChatMessage,
    ChatMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.pydantic_v1 import BaseModel, SecretStr, root_validator
from langchain_core.tools import BaseTool
from langchain_core.utils import get_from_dict_or_env
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from langchain_google_genai._common import GoogleGenerativeAIError
from langchain_google_genai.llms import GoogleModelFamily, _BaseGoogleGenerativeAI


def _convert_fc_type(fc: Union[BaseTool, Type[BaseModel]]) -> Dict:
    """
        Produce

        {
      "name": "get_weather",
      "description": "Determine weather in my location",
      "parameters": {
        "properties": {
          "location": {
            "description": "The city and state e.g. San Francisco, CA",
            "type_": 1
          },
          "unit": { "enum": ["c", "f"], "type_": 1 }
        },
        "required": ["location"],
        "type_": 6
      }
    }

    """
    if isinstance(fc, BaseTool):
        

    # type_: "Type"
    # format_: str
    # description: str
    # nullable: bool
    # enum: MutableSequence[str]
    # items: "Schema"
    # properties: MutableMapping[str, "Schema"]
    # required: MutableSequence[str]
    if "parameters" in fc:
        fc["parameters"] = _convert_fc_type(fc["parameters"])
    if "properties" in fc:
        for k, v in fc["properties"].items():
            fc["properties"][k] = _convert_fc_type(v)
    if "type" in fc:
        # STRING = 1
        # NUMBER = 2
        # INTEGER = 3
        # BOOLEAN = 4
        # ARRAY = 5
        # OBJECT = 6
        if fc["type"] == "string":
            fc["type_"] = 1
        elif fc["type"] == "number":
            fc["type_"] = 2
        elif fc["type"] == "integer":
            fc["type_"] = 3
        elif fc["type"] == "boolean":
            fc["type_"] = 4
        elif fc["type"] == "array":
            fc["type_"] = 5
        elif fc["type"] == "object":
            fc["type_"] = 6
        del fc["type"]
    if "format" in fc:
        fc["format_"] = fc["format"]
        del fc["format"]

    for k, v in fc.items():
        if isinstance(v, dict):
            fc[k] = _convert_fc_type(v)

    return fc