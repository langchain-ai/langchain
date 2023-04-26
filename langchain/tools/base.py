"""Base implementation for tools or skills."""
from __future__ import annotations

import inspect
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Sequence, Tuple, Type, Union

from pydantic import BaseModel

from langchain.tools.structured import BaseStructuredTool


class StringSchema(BaseModel):
    """Schema for a tool with string input."""

    # Child tools can add additional validation by
    # subclassing this schema.
    tool_input: str


class BaseTool(ABC, BaseStructuredTool[str]):
    """Interface for LangChain tools mapping a single string to an output string."""

    args_schema: Type[StringSchema] = StringSchema  # :meta private:

    def __init_subclass__(self) -> None:
        """Raise a warning if self._run's signature isn't compatible."""
        sig = inspect.signature(self._run)
        # Remove the 'self' argument but check others
        sig = sig.replace(
            parameters=[
                param for param in sig.parameters.values() if param.name != "self"
            ]
        )
        # Raise an error if there is more than one arg or if that arg isn't a string
        if len(sig.parameters) != 1:
            msg = f"""\
BaseTool's `_run` signature is expected to be 'def _run(tool_input: str) -> str'
but received {sig}
Please subclass the "BaseStructuredTool" for more complex run methods.

Examples:
--------

>>> from langchain.tools import BaseStructuredTool

>>> class {self.__name__}(BaseStructuredTool):
        ...
        def _run(self, ...) -> Any:
        ...

or use the StructuredTool decorator

>>> from langchain.tools import StructuredTool, structured_tool

>>> def {self.__name__.lower()}(arg1: int, arg2: dict) -> dict:
        ...

>>> tool = StructuredTool.from_function(my_tool)

Or equivalently

>>> @structured_tool
>>> def {self.__name__.lower()}(arg1: int, arg2: dict) -> dict:
       ...

"""
            warnings.warn(msg, DeprecationWarning)

    def _wrap_input(self, tool_input: Union[str, Dict]) -> Dict:
        """Wrap the tool's input into a pydantic model."""
        if isinstance(tool_input, Dict):
            return tool_input
        return {"tool_input": tool_input}

    def _prepare_input(self, input_: Dict) -> Tuple[Sequence, Dict]:
        """Prepare the args and kwargs for the tool."""
        # We expect a single string input
        return tuple(input_.values()), {}

    @abstractmethod
    def _run(self, tool_input: str) -> str:
        """Use the tool."""

    @abstractmethod
    async def _arun(self, tool_input: str) -> str:
        """Use the tool asynchronously."""

    def run(
        self,
        tool_input: Union[str, Dict],
        verbose: Optional[bool] = None,
        start_color: Optional[str] = "green",
        color: Optional[str] = "green",
        **kwargs: Any,
    ) -> str:
        """Use the tool."""
        wrapped_input = self._wrap_input(tool_input)
        return super().run(
            wrapped_input,
            verbose=verbose,
            start_color=start_color,
            color=color,
            **kwargs,
        )

    async def arun(
        self,
        tool_input: Union[str, Dict],
        verbose: Optional[bool] = None,
        start_color: Optional[str] = "green",
        color: Optional[str] = "green",
        **kwargs: Any,
    ) -> str:
        """Use the tool asynchronously."""
        wrapped_input = self._wrap_input(tool_input)
        return await super().arun(
            wrapped_input,
            verbose=verbose,
            start_color=start_color,
            color=color,
            **kwargs,
        )

    def __call__(self, tool_input: Union[str, Dict]) -> str:
        return self.run(tool_input)
