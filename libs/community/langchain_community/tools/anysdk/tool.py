import json
from json import JSONDecodeError
from typing import Any, Optional, Union

# ruff: ignore F811
from langchain_core.callbacks import (
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool


class AnySDKTool(BaseTool):
    """Tool for whatever function is passed into AnySDK."""

    client: Any
    name: str
    description: str

    def _run(
        self,
        tool_input: Union[str, dict, None] = None,
        *args: tuple,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: dict,
    ) -> str:
        try:
            if isinstance(tool_input, dict):
                params = tool_input
            elif isinstance(tool_input, str):
                try:
                    params = json.loads(tool_input)
                except JSONDecodeError:
                    params = {}
            else:
                params = {}

            func = getattr(self.client["client"], self.name)
            result = func(*args, **params, **kwargs)
            return json.dumps(result, default=str)
        except AttributeError:
            return f"Invalid function name: {self.name}"

    async def _arun(
        self,
        tool_input: Union[str, dict, None] = None,
        *args: tuple,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: dict,
    ) -> str:
        return self._run(
            tool_input,
            *args,
            run_manager=run_manager,
            **kwargs,
        )
