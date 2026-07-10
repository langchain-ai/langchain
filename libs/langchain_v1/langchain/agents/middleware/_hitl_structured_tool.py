"""StructuredTool variant that does not treat GraphInterrupt as a tool error.

LangGraph documents calling ``interrupt()`` inside a tool. That raises
``GraphInterrupt``, which subclasses ``Exception``. LangChain's ``BaseTool.run`` /
``arun`` invoke ``on_tool_error`` for any ``Exception`` before re-raising.

This helper re-raises ``GraphInterrupt`` before the broad error handler so human-
in-the-loop pauses are not reported as tool failures. See langgraph #8218.
"""

from __future__ import annotations

import uuid
from inspect import signature
from typing import Any

from langchain_core.callbacks import (
    AsyncCallbackManager,
    CallbackManager,
    Callbacks,
)
from langchain_core.runnables import RunnableConfig, patch_config
from langchain_core.runnables.config import set_config_context
from langchain_core.runnables.utils import coro_with_context
from langchain_core.tools.base import (
    BaseTool,
    ToolException,
    _format_output,
    _get_runnable_config_param,
    _handle_tool_error,
    _handle_validation_error,
)
from langchain_core.tools.structured import StructuredTool
from langgraph.errors import GraphInterrupt
from pydantic import ValidationError
from pydantic.v1 import ValidationError as ValidationErrorV1


class HITLStructuredTool(StructuredTool):
    """Like ``StructuredTool``, but does not signal ``on_tool_error`` for ``GraphInterrupt``."""

    def run(  # noqa: PLR0913
        self,
        tool_input: str | dict[str, Any],
        verbose: bool | None = None,  # noqa: FBT001
        start_color: str | None = "green",
        color: str | None = "green",
        callbacks: Callbacks = None,
        *,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        run_name: str | None = None,
        run_id: uuid.UUID | None = None,
        config: RunnableConfig | None = None,
        tool_call_id: str | None = None,
        **kwargs: Any,
    ) -> Any:
        callback_manager = CallbackManager.configure(
            callbacks,
            self.callbacks,
            self.verbose or bool(verbose),
            tags,
            self.tags,
            metadata,
            self.metadata,
        )

        filtered_tool_input = (
            self._filter_injected_args(tool_input) if isinstance(tool_input, dict) else None
        )

        tool_input_str = (
            tool_input
            if isinstance(tool_input, str)
            else str(filtered_tool_input if filtered_tool_input is not None else tool_input)
        )

        run_manager = callback_manager.on_tool_start(
            {"name": self.name, "description": self.description},
            tool_input_str,
            color=start_color,
            name=run_name,
            run_id=run_id,
            inputs=filtered_tool_input,
            tool_call_id=tool_call_id,
            **kwargs,
        )

        content = None
        artifact = None
        status = "success"
        try:
            child_config = patch_config(config, callbacks=run_manager.get_child())
            with set_config_context(child_config) as context:
                tool_args, tool_kwargs = self._to_args_and_kwargs(tool_input, tool_call_id)
                if signature(self._run).parameters.get("run_manager"):
                    tool_kwargs |= {"run_manager": run_manager}
                config_param = _get_runnable_config_param(self._run)
                if config_param:
                    tool_kwargs |= {config_param: config}
                response = context.run(self._run, *tool_args, **tool_kwargs)
            if self.response_format == "content_and_artifact":
                msg = (
                    "Since response_format='content_and_artifact' "
                    "a two-tuple of the message content and raw tool output is "
                    f"expected. Instead, generated response is of type: "
                    f"{type(response)}."
                )
                if not isinstance(response, tuple):
                    raise ValueError(msg)
                content, artifact = response
            else:
                content = response
        except (ValidationError, ValidationErrorV1) as e:
            if not self.handle_validation_error:
                run_manager.on_tool_error(e, tool_call_id=tool_call_id)
                raise
            content = _handle_validation_error(e, flag=self.handle_validation_error)
            status = "error"
        except ToolException as e:
            if not self.handle_tool_error:
                run_manager.on_tool_error(e, tool_call_id=tool_call_id)
                raise
            content = _handle_tool_error(e, flag=self.handle_tool_error)
            status = "error"
        except GraphInterrupt:
            raise
        except (Exception, KeyboardInterrupt) as e:
            run_manager.on_tool_error(e, tool_call_id=tool_call_id)
            raise
        output = _format_output(content, artifact, tool_call_id, self.name, status)
        run_manager.on_tool_end(output, color=color, name=self.name, **kwargs)
        return output

    async def arun(  # noqa: PLR0913
        self,
        tool_input: str | dict,
        verbose: bool | None = None,  # noqa: FBT001
        start_color: str | None = "green",
        color: str | None = "green",
        callbacks: Callbacks = None,
        *,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        run_name: str | None = None,
        run_id: uuid.UUID | None = None,
        config: RunnableConfig | None = None,
        tool_call_id: str | None = None,
        **kwargs: Any,
    ) -> Any:
        callback_manager = AsyncCallbackManager.configure(
            callbacks,
            self.callbacks,
            self.verbose or bool(verbose),
            tags,
            self.tags,
            metadata,
            self.metadata,
        )

        filtered_tool_input = (
            self._filter_injected_args(tool_input) if isinstance(tool_input, dict) else None
        )

        tool_input_str = (
            tool_input
            if isinstance(tool_input, str)
            else str(filtered_tool_input if filtered_tool_input is not None else tool_input)
        )

        run_manager = await callback_manager.on_tool_start(
            {"name": self.name, "description": self.description},
            tool_input_str,
            color=start_color,
            name=run_name,
            run_id=run_id,
            inputs=filtered_tool_input,
            tool_call_id=tool_call_id,
            **kwargs,
        )
        content = None
        artifact = None
        status = "success"
        try:
            tool_args, tool_kwargs = self._to_args_and_kwargs(tool_input, tool_call_id)
            child_config = patch_config(config, callbacks=run_manager.get_child())
            with set_config_context(child_config) as context:
                func_to_check = (
                    self._run
                    if self.__class__._arun is BaseTool._arun
                    else self._arun  # noqa: SLF001
                )
                if signature(func_to_check).parameters.get("run_manager"):
                    tool_kwargs["run_manager"] = run_manager
                config_param = _get_runnable_config_param(func_to_check)
                if config_param:
                    tool_kwargs[config_param] = config

                coro = self._arun(*tool_args, **tool_kwargs)
                response = await coro_with_context(coro, context)
            if self.response_format == "content_and_artifact":
                msg = (
                    "Since response_format='content_and_artifact' "
                    "a two-tuple of the message content and raw tool output is "
                    f"expected. Instead, generated response is of type: "
                    f"{type(response)}."
                )
                if not isinstance(response, tuple):
                    raise ValueError(msg)
                content, artifact = response
            else:
                content = response
        except ValidationError as e:
            if not self.handle_validation_error:
                await run_manager.on_tool_error(e, tool_call_id=tool_call_id)
                raise
            content = _handle_validation_error(e, flag=self.handle_validation_error)
            status = "error"
        except ToolException as e:
            if not self.handle_tool_error:
                await run_manager.on_tool_error(e, tool_call_id=tool_call_id)
                raise
            content = _handle_tool_error(e, flag=self.handle_tool_error)
            status = "error"
        except GraphInterrupt:
            raise
        except (Exception, KeyboardInterrupt) as e:
            await run_manager.on_tool_error(e, tool_call_id=tool_call_id)
            raise

        output = _format_output(content, artifact, tool_call_id, self.name, status)
        await run_manager.on_tool_end(output, color=color, name=self.name, **kwargs)
        return output
