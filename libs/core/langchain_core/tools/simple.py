"""Tool that takes in function or coroutine directly."""

from __future__ import annotations

from collections.abc import Awaitable
from inspect import signature
import time
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Optional,
    Union,
)

from typing_extensions import override

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.runnables import RunnableConfig, run_in_executor
from langchain_core.tools.base import (
    ArgsSchema,
    BaseTool,
    ToolException,
    _get_runnable_config_param,
)

if TYPE_CHECKING:
    from langchain_core.messages import ToolCall


class Tool(BaseTool):
    """用于简单操作的Tool实现。"""
    
    def _run(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """使用工具。"""
        if self.args_schema is not None and not isinstance(kwargs, dict):
            raise ValueError(
                "当提供args_schema时，kwargs必须是字典类型"
            )
        try:
            return self.func(*args, **kwargs)
        except Exception as e:
            error_msg = (
                f"{e.__class__.__name__} 在调用工具 {self.name} 时发生错误: {e}"
            )
            raise ToolException(error_msg) from e

    async def _arun(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """使用工具。"""
        if self.args_schema is not None and not isinstance(kwargs, dict):
            raise ValueError(
                "当提供args_schema时，kwargs必须是字典类型"
            )
        try:
            return await self.coroutine(*args, **kwargs)
        except Exception as e:
            error_msg = (
                f"{e.__class__.__name__} 在调用工具 {self.name} 时发生错误: {e}"
            )
            raise ToolException(error_msg) from e

    # --- Runnable ---

    @override
    async def ainvoke(
        self,
        input: Union[str, dict, ToolCall],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Any:
        if not self.coroutine:
            # If the tool does not implement async, fall back to default implementation
            return await run_in_executor(config, self.invoke, input, config, **kwargs)

        return await super().ainvoke(input, config, **kwargs)

    # --- Tool ---

    @property
    def args(self) -> dict:
        """The tool's input arguments.

        Returns:
            The input arguments for the tool.
        """
        if self.args_schema is not None:
            if isinstance(self.args_schema, dict):
                json_schema = self.args_schema
            else:
                json_schema = self.args_schema.model_json_schema()
            return json_schema["properties"]
        # For backwards compatibility, if the function signature is ambiguous,
        # assume it takes a single string input.
        return {"tool_input": {"type": "string"}}

    def _to_args_and_kwargs(
        self, tool_input: Union[str, dict], tool_call_id: Optional[str]
    ) -> tuple[tuple, dict]:
        """Convert tool input to pydantic model."""
        args, kwargs = super()._to_args_and_kwargs(tool_input, tool_call_id)
        # For backwards compatibility. The tool must be run with a single input
        all_args = list(args) + list(kwargs.values())
        if len(all_args) != 1:
            msg = (
                f"""Too many arguments to single-input tool {self.name}.
                Consider using StructuredTool instead."""
                f" Args: {all_args}"
            )
            raise ToolException(msg)
        return tuple(all_args), {}

    @override
    def _run(
        self,
        *args: Any,
        config: RunnableConfig,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> Any:
        """Use the tool."""
        
        # 增加消息优先级检查
        if hasattr(config, 'message_priority') and config.message_priority == 'low':
            if hasattr(self, 'throttle_low_priority'):
                time.sleep(self.throttle_low_priority)
            
        # 增加消息过期检查
        if hasattr(config, 'message_expiry') and config.message_expiry < time.time():
            raise ValueError("Message has expired")
            
        if self.func:
            if run_manager and signature(self.func).parameters.get("callbacks"):
                kwargs["callbacks"] = run_manager.get_child()
            if config_param := _get_runnable_config_param(self.func):
                kwargs[config_param] = config
            return self.func(*args, **kwargs)
        
        msg = "Tool does not support sync invocation."
        raise NotImplementedError(msg)

    async def _arun(
        self,
        *args: Any,
        config: RunnableConfig,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> Any:
        """Use the tool asynchronously."""
        if self.coroutine:
            if run_manager and signature(self.coroutine).parameters.get("callbacks"):
                kwargs["callbacks"] = run_manager.get_child()
            
            # 传递完整的执行上下文
            if run_manager and signature(self.coroutine).parameters.get("run_manager"):
                kwargs["run_manager"] = run_manager
            
            if config_param := _get_runnable_config_param(self.coroutine):
                kwargs[config_param] = config
            return await self.coroutine(*args, **kwargs)

        # NOTE: this code is unreachable since _arun is only called if coroutine is not
        # None.
        return await super()._arun(
            *args, config=config, run_manager=run_manager, **kwargs
        )

    # TODO: this is for backwards compatibility, remove in future
    def __init__(
        self, name: str, func: Optional[Callable], description: str, **kwargs: Any
    ) -> None:
        """Initialize tool."""
        super().__init__(name=name, func=func, description=description, **kwargs)

    @classmethod
    def from_function(
        cls,
        func: Optional[Callable],
        name: str,  # We keep these required to support backwards compatibility
        description: str,
        return_direct: bool = False,  # noqa: FBT001,FBT002
        args_schema: Optional[ArgsSchema] = None,
        coroutine: Optional[
            Callable[..., Awaitable[Any]]
        ] = None,  # This is last for compatibility, but should be after func
        **kwargs: Any,
    ) -> Tool:
        """Initialize tool from a function.

        Args:
            func: The function to create the tool from.
            name: The name of the tool.
            description: The description of the tool.
            return_direct: Whether to return the output directly. Defaults to False.
            args_schema: The schema of the tool's input arguments. Defaults to None.
            coroutine: The asynchronous version of the function. Defaults to None.
            kwargs: Additional arguments to pass to the tool.

        Returns:
            The tool.

        Raises:
            ValueError: If the function is not provided.
        """
        if func is None and coroutine is None:
            msg = "Function and/or coroutine must be provided"
            raise ValueError(msg)
        return cls(
            name=name,
            func=func,
            coroutine=coroutine,
            description=description,
            return_direct=return_direct,
            args_schema=args_schema,
            **kwargs,
        )
