from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Union

from pydantic import BaseModel

from langchain.tools.base import BaseTool


import re


def _parse_input(text) -> List[Union[str, int]]:
    """Parse input of the form data["key1"][0]["key2"] into a list of keys."""
    _res = re.findall(r'\[.*?]', text)
    # strip the brackets and quotes, convert to int if possible
    res = [i[1:-1].replace('"', '') for i in _res]
    res = [int(i) if i.isdigit() else i for i in res]
    return res


class JsonSpec(BaseModel):
    """Base class for JSON spec."""

    dict_: Dict[str, Any]
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
                    f"Value at path `{text}` is not a dict, use `value` instead to get its value directly."
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
            val = str(val)
            if len(val) > self.max_value_length:
                val = val[: self.max_value_length] + "..."
            return val
        except Exception as e:
            return repr(e)


class JsonSpecListKeysTool(BaseTool):
    """Tool for listing keys in a JSON spec."""

    name = "json_spec_list_keys"
    description = "Can be used to list all keys. Before calling this you should be SURE that the path to this exists."
    spec: JsonSpec

    def _run(self, tool_input: str) -> str:
        return self.spec.keys(tool_input)

    async def _arun(self, tool_input: str) -> str:
        return self._run(tool_input)


class JsonSpecGetValueTool(BaseTool):
    """Tool for getting a value in a JSON spec."""

    name = "json_spec_get_value"
    description = "Can be used to see value in string format. Before calling this you should be SURE that the path to this exists."
    spec: JsonSpec

    def _run(self, tool_input: str) -> str:
        return self.spec.value(tool_input)

    async def _arun(self, tool_input: str) -> str:
        return self._run(tool_input)
