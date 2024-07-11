"""**Tools** are classes that an Agent uses to interact with the world.

Each tool has a **description**. Agent uses the description to choose the right
tool for the job.

**Class hierarchy:**

.. code-block::

    RunnableSerializable --> BaseTool --> <name>Tool  # Examples: AIPluginTool, BaseGraphQLTool
                                          <name>      # Examples: BraveSearch, HumanInputRun

**Main helpers:**

.. code-block::

    CallbackManagerForToolRun, AsyncCallbackManagerForToolRun
"""  # noqa: E501

from __future__ import annotations

import asyncio
import inspect
import textwrap
import uuid
import warnings
from abc import ABC, abstractmethod
from contextvars import copy_context
from functools import partial
from inspect import signature
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    get_type_hints,
)

from typing_extensions import Annotated, get_args, get_origin

from langchain_core._api import deprecated
from langchain_core.callbacks import (
    AsyncCallbackManager,
    AsyncCallbackManagerForToolRun,
    BaseCallbackManager,
    CallbackManager,
    CallbackManagerForToolRun,
)
from langchain_core.callbacks.manager import (
    Callbacks,
)
from langchain_core.load.serializable import Serializable
from langchain_core.prompts import (
    BasePromptTemplate,
    PromptTemplate,
    aformat_document,
    format_document,
)
from langchain_core.pydantic_v1 import (
    BaseModel,
    Extra,
    Field,
    ValidationError,
    create_model,
    root_validator,
    validate_arguments,
)
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import (
    Runnable,
    RunnableConfig,
    RunnableSerializable,
    ensure_config,
)
from langchain_core.runnables.config import (
    _set_config_context,
    patch_config,
    run_in_executor,
)
from langchain_core.runnables.utils import accepts_context


class SchemaAnnotationError(TypeError):
    """Raised when 'args_schema' is missing or has an incorrect type annotation."""


def _is_annotated_type(typ: Type[Any]) -> bool:
    return get_origin(typ) is Annotated


def _get_annotation_description(arg: str, arg_type: Type[Any]) -> str | None:
    if _is_annotated_type(arg_type):
        annotated_args = get_args(arg_type)
        arg_type = annotated_args[0]
        if len(annotated_args) > 1:
            for annotation in annotated_args[1:]:
                if isinstance(annotation, str):
                    return annotation
    return None


def _create_subset_model(
    name: str,
    model: Type[BaseModel],
    field_names: list,
    *,
    descriptions: Optional[dict] = None,
    fn_description: Optional[str] = None,
) -> Type[BaseModel]:
    """Create a pydantic model with only a subset of model's fields."""
    fields = {}

    for field_name in field_names:
        field = model.__fields__[field_name]
        t = (
            # this isn't perfect but should work for most functions
            field.outer_type_
            if field.required and not field.allow_none
            else Optional[field.outer_type_]
        )
        if descriptions and field_name in descriptions:
            field.field_info.description = descriptions[field_name]
        fields[field_name] = (t, field.field_info)

    rtn = create_model(name, **fields)  # type: ignore
    rtn.__doc__ = textwrap.dedent(fn_description or model.__doc__ or "")
    return rtn


def _get_filtered_args(
    inferred_model: Type[BaseModel],
    func: Callable,
    *,
    filter_args: Sequence[str],
) -> dict:
    """Get the arguments from a function's signature."""
    schema = inferred_model.schema()["properties"]
    valid_keys = signature(func).parameters
    return {
        k: schema[k]
        for i, (k, param) in enumerate(valid_keys.items())
        if k not in filter_args and (i > 0 or param.name not in ("self", "cls"))
    }


def _parse_python_function_docstring(function: Callable) -> Tuple[str, dict]:
    """Parse the function and argument descriptions from the docstring of a function.

    Assumes the function docstring follows Google Python style guide.
    """
    docstring = inspect.getdoc(function)
    if docstring:
        docstring_blocks = docstring.split("\n\n")
        descriptors = []
        args_block = None
        past_descriptors = False
        for block in docstring_blocks:
            if block.startswith("Args:"):
                args_block = block
                break
            elif block.startswith("Returns:") or block.startswith("Example:"):
                # Don't break in case Args come after
                past_descriptors = True
            elif not past_descriptors:
                descriptors.append(block)
            else:
                continue
        description = " ".join(descriptors)
    else:
        description = ""
        args_block = None
    arg_descriptions = {}
    if args_block:
        arg = None
        for line in args_block.split("\n")[1:]:
            if ":" in line:
                arg, desc = line.split(":", maxsplit=1)
                arg_descriptions[arg.strip()] = desc.strip()
            elif arg:
                arg_descriptions[arg.strip()] += " " + line.strip()
    return description, arg_descriptions


def _infer_arg_descriptions(
    fn: Callable, *, parse_docstring: bool = False
) -> Tuple[str, dict]:
    """Infer argument descriptions from a function's docstring."""
    if parse_docstring:
        description, arg_descriptions = _parse_python_function_docstring(fn)
    else:
        description = inspect.getdoc(fn) or ""
        arg_descriptions = {}
    if hasattr(inspect, "get_annotations"):
        # This is for python < 3.10
        annotations = inspect.get_annotations(fn)  # type: ignore
    else:
        annotations = getattr(fn, "__annotations__", {})
    for arg, arg_type in annotations.items():
        if arg in arg_descriptions:
            continue
        if desc := _get_annotation_description(arg, arg_type):
            arg_descriptions[arg] = desc
    return description, arg_descriptions


class _SchemaConfig:
    """Configuration for the pydantic model."""

    extra: Any = Extra.forbid
    arbitrary_types_allowed: bool = True


def create_schema_from_function(
    model_name: str,
    func: Callable,
    *,
    filter_args: Optional[Sequence[str]] = None,
    parse_docstring: bool = False,
) -> Type[BaseModel]:
    """Create a pydantic schema from a function's signature.
    Args:
        model_name: Name to assign to the generated pydandic schema
        func: Function to generate the schema from
        filter_args: Optional list of arguments to exclude from the schema
        parse_docstring: Whether to parse the function's docstring for descriptions
             for each argument.
    Returns:
        A pydantic model with the same arguments as the function
    """
    # https://docs.pydantic.dev/latest/usage/validation_decorator/
    validated = validate_arguments(func, config=_SchemaConfig)  # type: ignore
    inferred_model = validated.model  # type: ignore
    filter_args = (
        filter_args if filter_args is not None else ("run_manager", "callbacks")
    )
    for arg in filter_args:
        if arg in inferred_model.__fields__:
            del inferred_model.__fields__[arg]
    description, arg_descriptions = _infer_arg_descriptions(
        func, parse_docstring=parse_docstring
    )
    # Pydantic adds placeholder virtual fields we need to strip
    valid_properties = _get_filtered_args(inferred_model, func, filter_args=filter_args)
    return _create_subset_model(
        f"{model_name}Schema",
        inferred_model,
        list(valid_properties),
        descriptions=arg_descriptions,
        fn_description=description,
    )


class ToolException(Exception):
    """Optional exception that tool throws when execution error occurs.

    When this exception is thrown, the agent will not stop working,
    but it will handle the exception according to the handle_tool_error
    variable of the tool, and the processing result will be returned
    to the agent as observation, and printed in red on the console.
    """

    pass


class BaseTool(RunnableSerializable[Union[str, Dict], Any]):
    """Interface LangChain tools must implement."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Create the definition of the new tool class."""
        super().__init_subclass__(**kwargs)

        args_schema_type = cls.__annotations__.get("args_schema", None)

        if args_schema_type is not None and args_schema_type == BaseModel:
            # Throw errors for common mis-annotations.
            # TODO: Use get_args / get_origin and fully
            # specify valid annotations.
            typehint_mandate = """
class ChildTool(BaseTool):
    ...
    args_schema: Type[BaseModel] = SchemaClass
    ..."""
            name = cls.__name__
            raise SchemaAnnotationError(
                f"Tool definition for {name} must include valid type annotations"
                f" for argument 'args_schema' to behave as expected.\n"
                f"Expected annotation of 'Type[BaseModel]'"
                f" but got '{args_schema_type}'.\n"
                f"Expected class looks like:\n"
                f"{typehint_mandate}"
            )

    name: str
    """The unique name of the tool that clearly communicates its purpose."""
    description: str
    """Used to tell the model how/when/why to use the tool.
    
    You can provide few-shot examples as a part of the description.
    """
    args_schema: Optional[Type[BaseModel]] = None
    """Pydantic model class to validate and parse the tool's input arguments."""
    return_direct: bool = False
    """Whether to return the tool's output directly. Setting this to True means
    
    that after the tool is called, the AgentExecutor will stop looping.
    """
    verbose: bool = False
    """Whether to log the tool's progress."""

    callbacks: Callbacks = Field(default=None, exclude=True)
    """Callbacks to be called during tool execution."""
    callback_manager: Optional[BaseCallbackManager] = Field(default=None, exclude=True)
    """Deprecated. Please use callbacks instead."""
    tags: Optional[List[str]] = None
    """Optional list of tags associated with the tool. Defaults to None
    These tags will be associated with each call to this tool,
    and passed as arguments to the handlers defined in `callbacks`.
    You can use these to eg identify a specific instance of a tool with its use case.
    """
    metadata: Optional[Dict[str, Any]] = None
    """Optional metadata associated with the tool. Defaults to None
    This metadata will be associated with each call to this tool,
    and passed as arguments to the handlers defined in `callbacks`.
    You can use these to eg identify a specific instance of a tool with its use case.
    """

    handle_tool_error: Optional[Union[bool, str, Callable[[ToolException], str]]] = (
        False
    )
    """Handle the content of the ToolException thrown."""

    handle_validation_error: Optional[
        Union[bool, str, Callable[[ValidationError], str]]
    ] = False
    """Handle the content of the ValidationError thrown."""

    class Config(Serializable.Config):
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    @property
    def is_single_input(self) -> bool:
        """Whether the tool only accepts a single input."""
        keys = {k for k in self.args if k != "kwargs"}
        return len(keys) == 1

    @property
    def args(self) -> dict:
        if self.args_schema is not None:
            return self.args_schema.schema()["properties"]
        else:
            schema = create_schema_from_function(self.name, self._run)
            return schema.schema()["properties"]

    # --- Runnable ---

    def get_input_schema(
        self, config: Optional[RunnableConfig] = None
    ) -> Type[BaseModel]:
        """The tool's input schema."""
        if self.args_schema is not None:
            return self.args_schema
        else:
            return create_schema_from_function(self.name, self._run)

    def invoke(
        self,
        input: Union[str, Dict],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Any:
        config = ensure_config(config)
        return self.run(
            input,
            callbacks=config.get("callbacks"),
            tags=config.get("tags"),
            metadata=config.get("metadata"),
            run_name=config.get("run_name"),
            run_id=config.pop("run_id", None),
            config=config,
            **kwargs,
        )

    async def ainvoke(
        self,
        input: Union[str, Dict],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Any:
        config = ensure_config(config)
        return await self.arun(
            input,
            callbacks=config.get("callbacks"),
            tags=config.get("tags"),
            metadata=config.get("metadata"),
            run_name=config.get("run_name"),
            run_id=config.pop("run_id", None),
            config=config,
            **kwargs,
        )

    # --- Tool ---

    def _parse_input(
        self,
        tool_input: Union[str, Dict],
    ) -> Union[str, Dict[str, Any]]:
        """Convert tool input to pydantic model."""
        input_args = self.args_schema
        if isinstance(tool_input, str):
            if input_args is not None:
                key_ = next(iter(input_args.__fields__.keys()))
                input_args.validate({key_: tool_input})
            return tool_input
        else:
            if input_args is not None:
                result = input_args.parse_obj(tool_input)
                return {
                    k: getattr(result, k)
                    for k, v in result.dict().items()
                    if k in tool_input
                }
        return tool_input

    @root_validator(pre=True)
    def raise_deprecation(cls, values: Dict) -> Dict:
        """Raise deprecation warning if callback_manager is used."""
        if values.get("callback_manager") is not None:
            warnings.warn(
                "callback_manager is deprecated. Please use callbacks instead.",
                DeprecationWarning,
            )
            values["callbacks"] = values.pop("callback_manager", None)
        return values

    @abstractmethod
    def _run(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Use the tool.

        Add run_manager: Optional[CallbackManagerForToolRun] = None
        to child implementations to enable tracing,
        """

    async def _arun(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Use the tool asynchronously.

        Add run_manager: Optional[AsyncCallbackManagerForToolRun] = None
        to child implementations to enable tracing,
        """
        return await run_in_executor(None, self._run, *args, **kwargs)

    def _to_args_and_kwargs(self, tool_input: Union[str, Dict]) -> Tuple[Tuple, Dict]:
        # For backwards compatibility, if run_input is a string,
        # pass as a positional argument.
        if isinstance(tool_input, str):
            return (tool_input,), {}
        else:
            return (), tool_input

    def run(
        self,
        tool_input: Union[str, Dict[str, Any]],
        verbose: Optional[bool] = None,
        start_color: Optional[str] = "green",
        color: Optional[str] = "green",
        callbacks: Callbacks = None,
        *,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        run_name: Optional[str] = None,
        run_id: Optional[uuid.UUID] = None,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Any:
        """Run the tool."""
        if not self.verbose and verbose is not None:
            verbose_ = verbose
        else:
            verbose_ = self.verbose
        callback_manager = CallbackManager.configure(
            callbacks,
            self.callbacks,
            verbose_,
            tags,
            self.tags,
            metadata,
            self.metadata,
        )
        # TODO: maybe also pass through run_manager is _run supports kwargs
        new_arg_supported = signature(self._run).parameters.get("run_manager")
        run_manager = callback_manager.on_tool_start(
            {"name": self.name, "description": self.description},
            tool_input if isinstance(tool_input, str) else str(tool_input),
            color=start_color,
            name=run_name,
            run_id=run_id,
            # Inputs by definition should always be dicts.
            # For now, it's unclear whether this assumption is ever violated,
            # but if it is we will send a `None` value to the callback instead
            # And will need to address issue via a patch.
            inputs=None if isinstance(tool_input, str) else tool_input,
            **kwargs,
        )
        try:
            child_config = patch_config(
                config,
                callbacks=run_manager.get_child(),
            )
            context = copy_context()
            context.run(_set_config_context, child_config)
            parsed_input = self._parse_input(tool_input)
            tool_args, tool_kwargs = self._to_args_and_kwargs(parsed_input)
            observation = (
                context.run(
                    self._run, *tool_args, run_manager=run_manager, **tool_kwargs
                )
                if new_arg_supported
                else context.run(self._run, *tool_args, **tool_kwargs)
            )
        except ValidationError as e:
            if not self.handle_validation_error:
                raise e
            elif isinstance(self.handle_validation_error, bool):
                observation = "Tool input validation error"
            elif isinstance(self.handle_validation_error, str):
                observation = self.handle_validation_error
            elif callable(self.handle_validation_error):
                observation = self.handle_validation_error(e)
            else:
                raise ValueError(
                    f"Got unexpected type of `handle_validation_error`. Expected bool, "
                    f"str or callable. Received: {self.handle_validation_error}"
                )
            return observation
        except ToolException as e:
            if not self.handle_tool_error:
                run_manager.on_tool_error(e)
                raise e
            elif isinstance(self.handle_tool_error, bool):
                if e.args:
                    observation = e.args[0]
                else:
                    observation = "Tool execution error"
            elif isinstance(self.handle_tool_error, str):
                observation = self.handle_tool_error
            elif callable(self.handle_tool_error):
                observation = self.handle_tool_error(e)
            else:
                raise ValueError(
                    f"Got unexpected type of `handle_tool_error`. Expected bool, str "
                    f"or callable. Received: {self.handle_tool_error}"
                )
            run_manager.on_tool_end(observation, color="red", name=self.name, **kwargs)
            return observation
        except (Exception, KeyboardInterrupt) as e:
            run_manager.on_tool_error(e)
            raise e
        else:
            run_manager.on_tool_end(observation, color=color, name=self.name, **kwargs)
            return observation

    async def arun(
        self,
        tool_input: Union[str, Dict],
        verbose: Optional[bool] = None,
        start_color: Optional[str] = "green",
        color: Optional[str] = "green",
        callbacks: Callbacks = None,
        *,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        run_name: Optional[str] = None,
        run_id: Optional[uuid.UUID] = None,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Any:
        """Run the tool asynchronously."""
        if not self.verbose and verbose is not None:
            verbose_ = verbose
        else:
            verbose_ = self.verbose
        callback_manager = AsyncCallbackManager.configure(
            callbacks,
            self.callbacks,
            verbose_,
            tags,
            self.tags,
            metadata,
            self.metadata,
        )
        new_arg_supported = signature(self._arun).parameters.get("run_manager")
        run_manager = await callback_manager.on_tool_start(
            {"name": self.name, "description": self.description},
            tool_input if isinstance(tool_input, str) else str(tool_input),
            color=start_color,
            name=run_name,
            inputs=tool_input,
            run_id=run_id,
            **kwargs,
        )
        try:
            parsed_input = self._parse_input(tool_input)
            # We then call the tool on the tool input to get an observation
            tool_args, tool_kwargs = self._to_args_and_kwargs(parsed_input)
            child_config = patch_config(
                config,
                callbacks=run_manager.get_child(),
            )
            context = copy_context()
            context.run(_set_config_context, child_config)
            coro = (
                context.run(
                    self._arun, *tool_args, run_manager=run_manager, **tool_kwargs
                )
                if new_arg_supported
                else context.run(self._arun, *tool_args, **tool_kwargs)
            )
            if accepts_context(asyncio.create_task):
                observation = await asyncio.create_task(coro, context=context)  # type: ignore
            else:
                observation = await coro

        except ValidationError as e:
            if not self.handle_validation_error:
                raise e
            elif isinstance(self.handle_validation_error, bool):
                observation = "Tool input validation error"
            elif isinstance(self.handle_validation_error, str):
                observation = self.handle_validation_error
            elif callable(self.handle_validation_error):
                observation = self.handle_validation_error(e)
            else:
                raise ValueError(
                    f"Got unexpected type of `handle_validation_error`. Expected bool, "
                    f"str or callable. Received: {self.handle_validation_error}"
                )
            return observation
        except ToolException as e:
            if not self.handle_tool_error:
                await run_manager.on_tool_error(e)
                raise e
            elif isinstance(self.handle_tool_error, bool):
                if e.args:
                    observation = e.args[0]
                else:
                    observation = "Tool execution error"
            elif isinstance(self.handle_tool_error, str):
                observation = self.handle_tool_error
            elif callable(self.handle_tool_error):
                observation = self.handle_tool_error(e)
            else:
                raise ValueError(
                    f"Got unexpected type of `handle_tool_error`. Expected bool, str "
                    f"or callable. Received: {self.handle_tool_error}"
                )
            await run_manager.on_tool_end(
                observation, color="red", name=self.name, **kwargs
            )
            return observation
        except (Exception, KeyboardInterrupt) as e:
            await run_manager.on_tool_error(e)
            raise e
        else:
            await run_manager.on_tool_end(
                observation, color=color, name=self.name, **kwargs
            )
            return observation

    @deprecated("0.1.47", alternative="invoke", removal="0.3.0")
    def __call__(self, tool_input: str, callbacks: Callbacks = None) -> str:
        """Make tool callable."""
        return self.run(tool_input, callbacks=callbacks)


class Tool(BaseTool):
    """Tool that takes in function or coroutine directly."""

    description: str = ""
    func: Optional[Callable[..., str]]
    """The function to run when the tool is called."""
    coroutine: Optional[Callable[..., Awaitable[str]]] = None
    """The asynchronous version of the function."""

    # --- Runnable ---

    async def ainvoke(
        self,
        input: Union[str, Dict],
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
        """The tool's input arguments."""
        if self.args_schema is not None:
            return self.args_schema.schema()["properties"]
        # For backwards compatibility, if the function signature is ambiguous,
        # assume it takes a single string input.
        return {"tool_input": {"type": "string"}}

    def _to_args_and_kwargs(self, tool_input: Union[str, Dict]) -> Tuple[Tuple, Dict]:
        """Convert tool input to pydantic model."""
        args, kwargs = super()._to_args_and_kwargs(tool_input)
        # For backwards compatibility. The tool must be run with a single input
        all_args = list(args) + list(kwargs.values())
        if len(all_args) != 1:
            raise ToolException(
                f"""Too many arguments to single-input tool {self.name}.
                Consider using StructuredTool instead."""
                f" Args: {all_args}"
            )
        return tuple(all_args), {}

    def _run(
        self,
        *args: Any,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> Any:
        """Use the tool."""
        if self.func:
            new_argument_supported = signature(self.func).parameters.get("callbacks")
            return (
                self.func(
                    *args,
                    callbacks=run_manager.get_child() if run_manager else None,
                    **kwargs,
                )
                if new_argument_supported
                else self.func(*args, **kwargs)
            )
        raise NotImplementedError("Tool does not support sync")

    async def _arun(
        self,
        *args: Any,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> Any:
        """Use the tool asynchronously."""
        if self.coroutine:
            new_argument_supported = signature(self.coroutine).parameters.get(
                "callbacks"
            )
            return (
                await self.coroutine(
                    *args,
                    callbacks=run_manager.get_child() if run_manager else None,
                    **kwargs,
                )
                if new_argument_supported
                else await self.coroutine(*args, **kwargs)
            )
        else:
            return await run_in_executor(
                None,
                self._run,
                run_manager=run_manager.get_sync() if run_manager else None,
                *args,
                **kwargs,
            )

    # TODO: this is for backwards compatibility, remove in future
    def __init__(
        self, name: str, func: Optional[Callable], description: str, **kwargs: Any
    ) -> None:
        """Initialize tool."""
        super(Tool, self).__init__(  # type: ignore[call-arg]
            name=name, func=func, description=description, **kwargs
        )

    @classmethod
    def from_function(
        cls,
        func: Optional[Callable],
        name: str,  # We keep these required to support backwards compatibility
        description: str,
        return_direct: bool = False,
        args_schema: Optional[Type[BaseModel]] = None,
        coroutine: Optional[
            Callable[..., Awaitable[Any]]
        ] = None,  # This is last for compatibility, but should be after func
        **kwargs: Any,
    ) -> Tool:
        """Initialize tool from a function."""
        if func is None and coroutine is None:
            raise ValueError("Function and/or coroutine must be provided")
        return cls(
            name=name,
            func=func,
            coroutine=coroutine,
            description=description,
            return_direct=return_direct,
            args_schema=args_schema,
            **kwargs,
        )


class StructuredTool(BaseTool):
    """Tool that can operate on any number of inputs."""

    description: str = ""
    args_schema: Type[BaseModel] = Field(..., description="The tool schema.")
    """The input arguments' schema."""
    func: Optional[Callable[..., Any]]
    """The function to run when the tool is called."""
    coroutine: Optional[Callable[..., Awaitable[Any]]] = None
    """The asynchronous version of the function."""

    # --- Runnable ---

    async def ainvoke(
        self,
        input: Union[str, Dict],
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
        """The tool's input arguments."""
        return self.args_schema.schema()["properties"]

    def _run(
        self,
        *args: Any,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> Any:
        """Use the tool."""
        if self.func:
            new_argument_supported = signature(self.func).parameters.get("callbacks")
            return (
                self.func(
                    *args,
                    callbacks=run_manager.get_child() if run_manager else None,
                    **kwargs,
                )
                if new_argument_supported
                else self.func(*args, **kwargs)
            )
        raise NotImplementedError("Tool does not support sync")

    async def _arun(
        self,
        *args: Any,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> str:
        """Use the tool asynchronously."""
        if self.coroutine:
            new_argument_supported = signature(self.coroutine).parameters.get(
                "callbacks"
            )
            return (
                await self.coroutine(
                    *args,
                    callbacks=run_manager.get_child() if run_manager else None,
                    **kwargs,
                )
                if new_argument_supported
                else await self.coroutine(*args, **kwargs)
            )
        return await run_in_executor(
            None,
            self._run,
            run_manager=run_manager.get_sync() if run_manager else None,
            *args,
            **kwargs,
        )

    @classmethod
    def from_function(
        cls,
        func: Optional[Callable] = None,
        coroutine: Optional[Callable[..., Awaitable[Any]]] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        return_direct: bool = False,
        args_schema: Optional[Type[BaseModel]] = None,
        infer_schema: bool = True,
        **kwargs: Any,
    ) -> StructuredTool:
        """Create tool from a given function.

        A classmethod that helps to create a tool from a function.

        Args:
            func: The function from which to create a tool
            coroutine: The async function from which to create a tool
            name: The name of the tool. Defaults to the function name
            description: The description of the tool. Defaults to the function docstring
            return_direct: Whether to return the result directly or as a callback
            args_schema: The schema of the tool's input arguments
            infer_schema: Whether to infer the schema from the function's signature
            **kwargs: Additional arguments to pass to the tool

        Returns:
            The tool

        Examples:

            .. code-block:: python

                def add(a: int, b: int) -> int:
                    \"\"\"Add two numbers\"\"\"
                    return a + b
                tool = StructuredTool.from_function(add)
                tool.run(1, 2) # 3
        """

        if func is not None:
            source_function = func
        elif coroutine is not None:
            source_function = coroutine
        else:
            raise ValueError("Function and/or coroutine must be provided")
        name = name or source_function.__name__
        description_ = description or source_function.__doc__
        if description_ is None and args_schema:
            description_ = args_schema.__doc__
        if description_ is None:
            raise ValueError(
                "Function must have a docstring if description not provided."
            )
        if description is None:
            # Only apply if using the function's docstring
            description_ = textwrap.dedent(description_).strip()

        # Description example:
        # search_api(query: str) - Searches the API for the query.
        description_ = f"{description_.strip()}"
        _args_schema = args_schema
        if _args_schema is None and infer_schema:
            # schema name is appended within function
            _args_schema = create_schema_from_function(name, source_function)
        return cls(
            name=name,
            func=func,
            coroutine=coroutine,
            args_schema=_args_schema,  # type: ignore[arg-type]
            description=description_,
            return_direct=return_direct,
            **kwargs,
        )


def tool(
    *args: Union[str, Callable, Runnable],
    return_direct: bool = False,
    args_schema: Optional[Type[BaseModel]] = None,
    infer_schema: bool = True,
) -> Callable:
    """Make tools out of functions, can be used with or without arguments.

    Args:
        *args: The arguments to the tool.
        return_direct: Whether to return directly from the tool rather
            than continuing the agent loop.
        args_schema: optional argument schema for user to specify
        infer_schema: Whether to infer the schema of the arguments from
            the function's signature. This also makes the resultant tool
            accept a dictionary input to its `run()` function.

    Requires:
        - Function must be of type (str) -> str
        - Function must have a docstring

    Examples:
        .. code-block:: python

            @tool
            def search_api(query: str) -> str:
                # Searches the API for the query.
                return

            @tool("search", return_direct=True)
            def search_api(query: str) -> str:
                # Searches the API for the query.
                return
    """

    def _make_with_name(tool_name: str) -> Callable:
        def _make_tool(dec_func: Union[Callable, Runnable]) -> BaseTool:
            if isinstance(dec_func, Runnable):
                runnable = dec_func

                if runnable.input_schema.schema().get("type") != "object":
                    raise ValueError("Runnable must have an object schema.")

                async def ainvoke_wrapper(
                    callbacks: Optional[Callbacks] = None, **kwargs: Any
                ) -> Any:
                    return await runnable.ainvoke(kwargs, {"callbacks": callbacks})

                def invoke_wrapper(
                    callbacks: Optional[Callbacks] = None, **kwargs: Any
                ) -> Any:
                    return runnable.invoke(kwargs, {"callbacks": callbacks})

                coroutine = ainvoke_wrapper
                func = invoke_wrapper
                schema: Optional[Type[BaseModel]] = runnable.input_schema
                description = repr(runnable)
            elif inspect.iscoroutinefunction(dec_func):
                coroutine = dec_func
                func = None
                schema = args_schema
                description = None
            else:
                coroutine = None
                func = dec_func
                schema = args_schema
                description = None

            if infer_schema or args_schema is not None:
                return StructuredTool.from_function(
                    func,
                    coroutine,
                    name=tool_name,
                    description=description,
                    return_direct=return_direct,
                    args_schema=schema,
                    infer_schema=infer_schema,
                )
            # If someone doesn't want a schema applied, we must treat it as
            # a simple string->string function
            if func.__doc__ is None:
                raise ValueError(
                    "Function must have a docstring if "
                    "description not provided and infer_schema is False."
                )
            return Tool(
                name=tool_name,
                func=func,
                description=f"{tool_name} tool",
                return_direct=return_direct,
                coroutine=coroutine,
            )

        return _make_tool

    if len(args) == 2 and isinstance(args[0], str) and isinstance(args[1], Runnable):
        return _make_with_name(args[0])(args[1])
    elif len(args) == 1 and isinstance(args[0], str):
        # if the argument is a string, then we use the string as the tool name
        # Example usage: @tool("search", return_direct=True)
        return _make_with_name(args[0])
    elif len(args) == 1 and callable(args[0]):
        # if the argument is a function, then we use the function name as the tool name
        # Example usage: @tool
        return _make_with_name(args[0].__name__)(args[0])
    elif len(args) == 0:
        # if there are no arguments, then we use the function name as the tool name
        # Example usage: @tool(return_direct=True)
        def _partial(func: Callable[[str], str]) -> BaseTool:
            return _make_with_name(func.__name__)(func)

        return _partial
    else:
        raise ValueError("Too many arguments for tool decorator")


class RetrieverInput(BaseModel):
    """Input to the retriever."""

    query: str = Field(description="query to look up in retriever")


def _get_relevant_documents(
    query: str,
    retriever: BaseRetriever,
    document_prompt: BasePromptTemplate,
    document_separator: str,
    callbacks: Callbacks = None,
) -> str:
    docs = retriever.invoke(query, config={"callbacks": callbacks})
    return document_separator.join(
        format_document(doc, document_prompt) for doc in docs
    )


async def _aget_relevant_documents(
    query: str,
    retriever: BaseRetriever,
    document_prompt: BasePromptTemplate,
    document_separator: str,
    callbacks: Callbacks = None,
) -> str:
    docs = await retriever.ainvoke(query, config={"callbacks": callbacks})
    return document_separator.join(
        [await aformat_document(doc, document_prompt) for doc in docs]
    )


def create_retriever_tool(
    retriever: BaseRetriever,
    name: str,
    description: str,
    *,
    document_prompt: Optional[BasePromptTemplate] = None,
    document_separator: str = "\n\n",
) -> Tool:
    """Create a tool to do retrieval of documents.

    Args:
        retriever: The retriever to use for the retrieval
        name: The name for the tool. This will be passed to the language model,
            so should be unique and somewhat descriptive.
        description: The description for the tool. This will be passed to the language
            model, so should be descriptive.

    Returns:
        Tool class to pass to an agent
    """
    document_prompt = document_prompt or PromptTemplate.from_template("{page_content}")
    func = partial(
        _get_relevant_documents,
        retriever=retriever,
        document_prompt=document_prompt,
        document_separator=document_separator,
    )
    afunc = partial(
        _aget_relevant_documents,
        retriever=retriever,
        document_prompt=document_prompt,
        document_separator=document_separator,
    )
    return Tool(
        name=name,
        description=description,
        func=func,
        coroutine=afunc,
        args_schema=RetrieverInput,
    )


ToolsRenderer = Callable[[List[BaseTool]], str]


def render_text_description(tools: List[BaseTool]) -> str:
    """Render the tool name and description in plain text.

    Output will be in the format of:

    .. code-block:: markdown

        search: This tool is used for search
        calculator: This tool is used for math
    """
    descriptions = []
    for tool in tools:
        if hasattr(tool, "func") and tool.func:
            sig = signature(tool.func)
            description = f"{tool.name}{sig} - {tool.description}"
        else:
            description = f"{tool.name} - {tool.description}"

        descriptions.append(description)
    return "\n".join(descriptions)


def render_text_description_and_args(tools: List[BaseTool]) -> str:
    """Render the tool name, description, and args in plain text.

    Output will be in the format of:

    .. code-block:: markdown

        search: This tool is used for search, args: {"query": {"type": "string"}}
        calculator: This tool is used for math, \
args: {"expression": {"type": "string"}}
    """
    tool_strings = []
    for tool in tools:
        args_schema = str(tool.args)
        if hasattr(tool, "func") and tool.func:
            sig = signature(tool.func)
            description = f"{tool.name}{sig} - {tool.description}"
        else:
            description = f"{tool.name} - {tool.description}"
        tool_strings.append(f"{description}, args: {args_schema}")
    return "\n".join(tool_strings)


class BaseToolkit(BaseModel, ABC):
    """Base Toolkit representing a collection of related tools."""

    @abstractmethod
    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""


def _get_description_from_runnable(runnable: Runnable) -> str:
    """Generate a placeholder description of a runnable."""
    input_schema = runnable.input_schema.schema()
    return f"Takes {input_schema}."


def _get_schema_from_runnable_and_arg_types(
    runnable: Runnable,
    name: str,
    arg_types: Optional[Dict[str, Type]] = None,
) -> Type[BaseModel]:
    """Infer args_schema for tool."""
    if arg_types is None:
        try:
            arg_types = get_type_hints(runnable.InputType)
        except TypeError as e:
            raise TypeError(
                "Tool input must be str or dict. If dict, dict arguments must be "
                "typed. Either annotate types (e.g., with TypedDict) or pass "
                f"arg_types into `.as_tool` to specify. {str(e)}"
            )
    fields = {key: (key_type, Field(...)) for key, key_type in arg_types.items()}
    return create_model(name, **fields)  # type: ignore


def convert_runnable_to_tool(
    runnable: Runnable,
    name: Optional[str] = None,
    description: Optional[str] = None,
    arg_types: Optional[Dict[str, Type]] = None,
) -> BaseTool:
    """Convert a Runnable into a BaseTool."""
    description = description or _get_description_from_runnable(runnable)
    name = name or runnable.get_name()

    schema = runnable.input_schema.schema()
    if schema.get("type") == "string":
        return Tool(
            name=name,
            func=runnable.invoke,
            coroutine=runnable.ainvoke,
            description=description,
        )
    else:

        async def ainvoke_wrapper(
            callbacks: Optional[Callbacks] = None, **kwargs: Any
        ) -> Any:
            return await runnable.ainvoke(kwargs, config={"callbacks": callbacks})

        def invoke_wrapper(callbacks: Optional[Callbacks] = None, **kwargs: Any) -> Any:
            return runnable.invoke(kwargs, config={"callbacks": callbacks})

        if (
            arg_types is None
            and schema.get("type") == "object"
            and schema.get("properties")
        ):
            args_schema = runnable.input_schema
        else:
            args_schema = _get_schema_from_runnable_and_arg_types(
                runnable, name, arg_types=arg_types
            )

        return StructuredTool.from_function(
            name=name,
            func=invoke_wrapper,
            coroutine=ainvoke_wrapper,
            description=description,
            args_schema=args_schema,
        )
