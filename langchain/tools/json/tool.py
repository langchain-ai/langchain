from __future__ import annotations
from pydantic import BaseModel
from pathlib import Path
from typing import Any, Dict, List, Union
import json
from langchain.tools.base import BaseTool

import re


def _parse_input(text: str) -> List[Union[str, int]]:
    _res = re.findall(r'\[.*?]', text)
    res = []
    for r in _res:
        val = r[1:-1]
        if val[0] != '"':
            val = int(val)
        else:
            val = val[1:-1]
        res.append(val)
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

    def keys(self, text: str):
        try:
            items = _parse_input(text)
            val = self.dict
            for i in items:
                val = val[i]
            return str(val.keys())
        except Exception as e:
            return repr(e)

    def value(self, text: str):
        try:
            items = _parse_input(text)
            val = self.dict
            for i in items:
                val = val[i]

            if isinstance(val, dict) and len(str(val)) > self.max_value_length:
                return "Value is a large dictionary, should explore its keys directly"
            val = str(val)
            if len(val) > self.max_value_length:
                val = val[self.max_value_length] + "..."
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
