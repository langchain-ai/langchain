"""Pipeline Tool"""
from __future__ import annotations

import json
from copy import copy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jmespath
from pydantic import BaseModel

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.tools.base import BaseTool


def _return_based_on_expression(
    output_expression: Any,
    out: Any,
    extra: Optional[Union[str, Dict[str, Any]]] = None,
    final: bool = False,
) -> Any:
    if output_expression is None:
        return out
    if extra:
        extra = copy(extra)

    if final:
        rval = {}
    else:
        rval = copy(extra) if isinstance(extra, dict) else {}

    if isinstance(output_expression, str):
        rval[output_expression] = out
    elif isinstance(output_expression, (tuple, list)):
        for k in output_expression:
            rval[k] = out[k]
    else:
        for k, expression in output_expression.items():
            rval[k] = jmespath.search(expression, out)

    if final and len(rval) == 1:
        return list(rval.values())[0]

    return rval


class PipelineStep(BaseModel):
    """Base class for a pipeline step."""

    func: Callable
    input_expression: Optional[Union[str, Dict[str, str]]] = None
    output_expression: Optional[Union[str, List[str], Dict[str, str]]] = None

    def __call__(
        self, input_: Union[str, Dict[str, Any]]
    ) -> Union[str, Dict[str, Any]]:
        """Execute the step function.

        Args:
            input_ (Union[str, Dict[str, Any]]): The input to the step function.

        Raises:
            ValueError: Raised when there is a mismatch between the input and input_expression.

        Returns:
            Union[str, Dict[str, Any]]: The extracted output based on the output_expression.
        """
        if self.input_expression is not None:
            if isinstance(input_, str):
                try:
                    input_ = json.loads(input_)
                except json.JSONDecodeError:
                    pass

            if not isinstance(input_, dict):
                raise ValueError(
                    "Input value should be a dict if input_expression is available"
                )

            args, kw = [], {}
            if isinstance(self.input_expression, dict):
                for key, expression in self.input_expression.items():
                    value = jmespath.search(expression, input_)
                    kw[key] = value
            else:
                kw[self.input_expression] = input_[self.input_expression]
        if not self.input_expression:
            args, kw = [input_], {}

        out = self.func(*args, **kw)
        return _return_based_on_expression(self.output_expression, out, extra=input_)


class PipelineTool(BaseTool):
    """The Pipeline Tool is designed to facilitate the creation of pipelines.

    It allows for the utilization of existing agents, chains, and tools
    within each step of the pipeline.
    """

    name = "pipeline_tool"
    description = """
    Can be used to construct a context task handler with existing tools and chains.
    """

    steps: List[PipelineStep]
    output_expression: Optional[Union[str, List[str], Dict[str, str]]] = None

    def _to_args_and_kwargs(self, tool_input: str | Dict) -> Tuple[Tuple, Dict]:
        return (tool_input,), {}

    def _run(
        self,
        tool_input: Union[str, Dict],
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        input_next = self._parse_input(tool_input)
        for step in self.steps:
            out = step(input_next)
            input_next = out
        out = input_next
        out = _return_based_on_expression(self.output_expression, out, final=True)
        return json.dumps(out)

    async def _arun(
        self,
        tool_input: Union[str, Dict],
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        return self._run(tool_input)
