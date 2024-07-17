# flake8: noqa
"""Tools for working with JSON specs."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Union

from langchain_core.pydantic_v1 import BaseModel

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool


def _parse_input(text: str) -> List[Union[str, int]]:
    """Parse input of the form data["key1"][0]["key2"] into a list of keys."""
    _res = re.findall(r"\[.*?]", text)
    # strip the brackets and quotes, convert to int if possible
    res = [i[1:-1].replace('"', "").replace("'", "") for i in _res]
    res = [int(i) if i.isdigit() else i for i in res]
    return res


class JsonSpec(BaseModel):
    """Base class for JSON spec."""

    dict_: Dict
    max_value_length: int = 200

    @classmethod
    def from_file(cls, path: Path) -> JsonSpec:
        """Create a JsonSpec from a file."""
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        dict_ = json.loads(path.read_text())
        return cls(dict_=dict_)

    def keys(self, text: str) -> str:
        """Return the keys of the dict at the given path.

        Args:
            text: Python representation of the path to the dict (e.g. data["key1"][0]["key2"]).
        """
        try:
            items = _parse_input(text)
            val = self.dict_
            for i in items:
                if i:
                    val = val[i]
            if not isinstance(val, dict):
                raise ValueError(
                    f"Value at path `{text}` is not a dict, get the value directly."
                )
            return str(list(val.keys()))
        except Exception as e:
            return repr(e)

    def value(self, text: str) -> str:
        """Return the value of the dict at the given path.

        Args:
            text: Python representation of the path to the dict (e.g. data["key1"][0]["key2"]).
        """
        try:
            items = _parse_input(text)
            val = self.dict_
            for i in items:
                val = val[i]

            if isinstance(val, dict) and len(str(val)) > self.max_value_length:
                return "Value is a large dictionary, should explore its keys directly"
            str_val = str(val)
            if len(str_val) > self.max_value_length:
                str_val = str_val[: self.max_value_length] + "..."
            return str_val
        except Exception as e:
            return repr(e)


class JsonListKeysTool(BaseTool):
    """Tool for listing keys in a JSON spec."""

    name: str = "json_spec_list_keys"
    description: str = """
    Can be used to list all keys at a given path. 
    Before calling this you should be SURE that the path to this exists.
    The input is a text representation of the path to the dict in Python syntax (e.g. data["key1"][0]["key2"]).
    """
    spec: JsonSpec

    def _run(
        self,
        tool_input: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        return self.spec.keys(tool_input)

    async def _arun(
        self,
        tool_input: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        return self._run(tool_input)


class JsonGetValueTool(BaseTool):
    """Tool for getting a value in a JSON spec."""

    name: str = "json_spec_get_value"
    description: str = """
    Can be used to see value in string format at a given path.
    Before calling this you should be SURE that the path to this exists.
    The input is a text representation of the path to the dict in Python syntax (e.g. data["key1"][0]["key2"]).
    """
    spec: JsonSpec

    def _run(
        self,
        tool_input: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        return self.spec.value(tool_input)

    async def _arun(
        self,
        tool_input: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        return self._run(tool_input)
