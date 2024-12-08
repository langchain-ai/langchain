from __future__ import annotations

import asyncio
import functools
import inspect
import json
import uuid
import warnings
from abc import ABC, abstractmethod
from collections.abc import Sequence
from contextvars import copy_context
from inspect import signature
from typing import (
    Annotated,
    Any,
    Callable,
    Literal,
    Optional,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PydanticDeprecationWarning,
    SkipValidation,
    ValidationError,
    model_validator,
    validate_arguments,
)
from pydantic.v1 import BaseModel as BaseModelV1
from pydantic.v1 import ValidationError as ValidationErrorV1
from pydantic.v1 import validate_arguments as validate_arguments_v1

from langchain_core._api import deprecated
from langchain_core.callbacks import (
    AsyncCallbackManager,
    BaseCallbackManager,
    CallbackManager,
    Callbacks,
)
from langchain_core.messages.tool import ToolCall, ToolMessage
from langchain_core.runnables import (
    RunnableConfig,
    RunnableSerializable,
    ensure_config,
    patch_config,
    run_in_executor,
)
from langchain_core.runnables.config import _set_config_context
from langchain_core.runnables.utils import asyncio_accepts_context
from langchain_core.utils.function_calling import (
    _parse_google_docstring,
    _py_38_safe_origin,
)
from langchain_core.utils.pydantic import (
    TypeBaseModel,
    _create_subset_model,
    get_fields,
    is_basemodel_subclass,
    is_pydantic_v1_subclass,
    is_pydantic_v2_subclass,
)

FILTERED_ARGS = ("run_manager", "callbacks")


class SchemaAnnotationError(TypeError):
    """Raised when 'args_schema' is missing or has an incorrect type annotation."""


def _is_annotated_type(typ: type[Any]) -> bool:
    return get_origin(typ) is Annotated


def _get_annotation_description(arg_type: type) -> str | None:
    if _is_annotated_type(arg_type):
        annotated_args = get_args(arg_type)
        for annotation in annotated_args[1:]:
            if isinstance(annotation, str):
                return annotation
    return None


def _get_filtered_args(
    inferred_model: type[BaseModel],
    func: Callable,
    *,
    filter_args: Sequence[str],
    include_injected: bool = True,
) -> dict:
    """Get the arguments from a function's signature."""
    schema = inferred_model.model_json_schema()["properties"]
    valid_keys = signature(func).parameters
    return {
        k: schema[k]
        for i, (k, param) in enumerate(valid_keys.items())
        if k not in filter_args
        and (i > 0 or param.name not in ("self", "cls"))
        and (include_injected or not _is_injected_arg_type(param.annotation))
    }


def _parse_python_function_docstring(
    function: Callable, annotations: dict, error_on_invalid_docstring: bool = False
) -> tuple[str, dict]:
    """Parse the function and argument descriptions from the docstring of a function.

    Assumes the function docstring follows Google Python style guide.
    """
    docstring = inspect.getdoc(function)
    return _parse_google_docstring(
        docstring,
        list(annotations),
        error_on_invalid_docstring=error_on_invalid_docstring,
    )


def _validate_docstring_args_against_annotations(
    arg_descriptions: dict, annotations: dict
) -> None:
    """Raise error if docstring arg is not in type annotations."""
    for docstring_arg in arg_descriptions:
        if docstring_arg not in annotations:
            msg = f"Arg {docstring_arg} in docstring not found in function signature."
            raise ValueError(msg)


def _infer_arg_descriptions(
    fn: Callable,
    *,
    parse_docstring: bool = False,
    error_on_invalid_docstring: bool = False,
) -> tuple[str, dict]:
    """Infer argument descriptions from a function's docstring."""
    if hasattr(inspect, "get_annotations"):
        # This is for python < 3.10
        annotations = inspect.get_annotations(fn)  # type: ignore
    else:
        annotations = getattr(fn, "__annotations__", {})
    if parse_docstring:
        description, arg_descriptions = _parse_python_function_docstring(
            fn, annotations, error_on_invalid_docstring=error_on_invalid_docstring
        )
    else:
        description = inspect.getdoc(fn) or ""
        arg_descriptions = {}
    if parse_docstring:
        _validate_docstring_args_against_annotations(arg_descriptions, annotations)
    for arg, arg_type in annotations.items():
        if arg in arg_descriptions:
            continue
        if desc := _get_annotation_description(arg_type):
            arg_descriptions[arg] = desc
    return description, arg_descriptions


def _is_pydantic_annotation(annotation: Any, pydantic_version: str = "v2") -> bool:
    """Determine if a type annotation is a Pydantic model."""
    base_model_class = BaseModelV1 if pydantic_version == "v1" else BaseModel
    try:
        return issubclass(annotation, base_model_class)
    except TypeError:
        return False


def _function_annotations_are_pydantic_v1(
    signature: inspect.Signature, func: Callable
) -> bool:
    """Determine if all Pydantic annotations in a function signature are from V1."""
    any_v1_annotations = any(
        _is_pydantic_annotation(parameter.annotation, pydantic_version="v1")
        for parameter in signature.parameters.values()
    )
    any_v2_annotations = any(
        _is_pydantic_annotation(parameter.annotation, pydantic_version="v2")
        for parameter in signature.parameters.values()
    )
    if any_v1_annotations and any_v2_annotations:
        msg = (
            f"Function {func} contains a mix of Pydantic v1 and v2 annotations. "
            "Only one version of Pydantic annotations per function is supported."
        )
        raise NotImplementedError(msg)
    return any_v1_annotations and not any_v2_annotations


class _SchemaConfig:
    """Configuration for the pydantic model.

    This is used to configure the pydantic model created from
    a function's signature.

    Parameters:
        extra: Whether to allow extra fields in the model.
        arbitrary_types_allowed: Whether to allow arbitrary types in the model.
            Defaults to True.
    """

    extra: str = "forbid"
    arbitrary_types_allowed: bool = True


def create_schema_from_function(
    model_name: str,
    func: Callable,
    *,
    filter_args: Optional[Sequence[str]] = None,
    parse_docstring: bool = False,
    error_on_invalid_docstring: bool = False,
    include_injected: bool = True,
) -> type[BaseModel]:
    """Create a pydantic schema from a function's signature.

    Args:
        model_name: Name to assign to the generated pydantic schema.
        func: Function to generate the schema from.
        filter_args: Optional list of arguments to exclude from the schema.
            Defaults to FILTERED_ARGS.
        parse_docstring: Whether to parse the function's docstring for descriptions
            for each argument. Defaults to False.
        error_on_invalid_docstring: if ``parse_docstring`` is provided, configure
            whether to raise ValueError on invalid Google Style docstrings.
            Defaults to False.
        include_injected: Whether to include injected arguments in the schema.
            Defaults to True, since we want to include them in the schema
            when *validating* tool inputs.

    Returns:
        A pydantic model with the same arguments as the function.
    """
    sig = inspect.signature(func)

    if _function_annotations_are_pydantic_v1(sig, func):
        validated = validate_arguments_v1(func, config=_SchemaConfig)  # type: ignore
    else:
        # https://docs.pydantic.dev/latest/usage/validation_decorator/
        with warnings.catch_warnings():
            # We are using deprecated functionality here.
            # This code should be re-written to simply construct a pydantic model
            # using inspect.signature and create_model.
            warnings.simplefilter("ignore", category=PydanticDeprecationWarning)
            validated = validate_arguments(func, config=_SchemaConfig)  # type: ignore

    # Let's ignore `self` and `cls` arguments for class and instance methods
    # If qualified name has a ".", then it likely belongs in a class namespace
    in_class = bool(func.__qualname__ and "." in func.__qualname__)

    has_args = False
    has_kwargs = False

    for param in sig.parameters.values():
        if param.kind == param.VAR_POSITIONAL:
            has_args = True
        elif param.kind == param.VAR_KEYWORD:
            has_kwargs = True

    inferred_model = validated.model  # type: ignore

    if filter_args:
        filter_args_ = filter_args
    else:
        # Handle classmethods and instance methods
        existing_params: list[str] = list(sig.parameters.keys())
        if existing_params and existing_params[0] in ("self", "cls") and in_class:
            filter_args_ = [existing_params[0]] + list(FILTERED_ARGS)
        else:
            filter_args_ = list(FILTERED_ARGS)

        for existing_param in existing_params:
            if not include_injected and _is_injected_arg_type(
                sig.parameters[existing_param].annotation
            ):
                filter_args_.append(existing_param)

    description, arg_descriptions = _infer_arg_descriptions(
        func,
        parse_docstring=parse_docstring,
        error_on_invalid_docstring=error_on_invalid_docstring,
    )
    # Pydantic adds placeholder virtual fields we need to strip
    valid_properties = []
    for field in get_fields(inferred_model):
        if not has_args and field == "args":
            continue
        if not has_kwargs and field == "kwargs":
            continue

        if field == "v__duplicate_kwargs":  # Internal pydantic field
            continue

        if field not in filter_args_:
            valid_properties.append(field)

    return _create_subset_model(
        model_name,
        inferred_model,
        list(valid_properties),
        descriptions=arg_descriptions,
        fn_description=description,
    )


class ToolException(Exception):  # noqa: N818
    """Optional exception that tool throws when execution error occurs.

    When this exception is thrown, the agent will not stop working,
    but it will handle the exception according to the handle_tool_error
    variable of the tool, and the processing result will be returned
    to the agent as observation, and printed in red on the console.
    """


class BaseTool(RunnableSerializable[Union[str, dict, ToolCall], Any]):
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
            msg = (
                f"Tool definition for {name} must include valid type annotations"
                f" for argument 'args_schema' to behave as expected.\n"
                f"Expected annotation of 'Type[BaseModel]'"
                f" but got '{args_schema_type}'.\n"
                f"Expected class looks like:\n"
                f"{typehint_mandate}"
            )
            raise SchemaAnnotationError(msg)

    name: str
    """The unique name of the tool that clearly communicates its purpose."""
    description: Optional[str] = None
    """Used to tell the model how/when/why to use the tool.

    You can provide few-shot examples as a part of the description.
    """

    args_schema: Annotated[Optional[TypeBaseModel], SkipValidation()] = Field(
        default=None, description="The tool schema."
    )
    """Pydantic model class to validate and parse the tool's input arguments.

    Args schema should be either:

    - A subclass of pydantic.BaseModel.
    or
    - A subclass of pydantic.v1.BaseModel if accessing v1 namespace in pydantic 2
    """
    return_direct: bool = False
    """Whether to return the tool's output directly.

    Setting this to True means
    that after the tool is called, the AgentExecutor will stop looping.
    """
    verbose: bool = False
    """Whether to log the tool's progress."""

    callbacks: Callbacks = Field(default=None, exclude=True)
    """Callbacks to be called during tool execution."""

    callback_manager: Optional[BaseCallbackManager] = deprecated(
        name="callback_manager", since="0.1.7", removal="1.0", alternative="callbacks"
    )(
        Field(
            default=None,
            exclude=True,
            description="Callback manager to add to the run trace.",
        )
    )
    tags: Optional[list[str]] = None
    """Optional list of tags associated with the tool. Defaults to None.
    These tags will be associated with each call to this tool,
    and passed as arguments to the handlers defined in `callbacks`.
    You can use these to eg identify a specific instance of a tool with its use case.
    """
    metadata: Optional[dict[str, Any]] = None
    """Optional metadata associated with the tool. Defaults to None.
    This metadata will be associated with each call to this tool,
    and passed as arguments to the handlers defined in `callbacks`.
    You can use these to eg identify a specific instance of a tool with its use case.
    """

    handle_tool_error: Optional[Union[bool, str, Callable[[ToolException], str]]] = (
        False
    )
    """Handle the content of the ToolException thrown."""

    handle_validation_error: Optional[
        Union[bool, str, Callable[[Union[ValidationError, ValidationErrorV1]], str]]
    ] = False
    """Handle the content of the ValidationError thrown."""

    response_format: Literal["content", "content_and_artifact"] = "content"
    """The tool response format. Defaults to 'content'.

    If "content" then the output of the tool is interpreted as the contents of a
    ToolMessage. If "content_and_artifact" then the output is expected to be a
    two-tuple corresponding to the (content, artifact) of a ToolMessage.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the tool."""
        if (
            "args_schema" in kwargs
            and kwargs["args_schema"] is not None
            and not is_basemodel_subclass(kwargs["args_schema"])
        ):
            msg = (
                f"args_schema must be a subclass of pydantic BaseModel. "
                f"Got: {kwargs['args_schema']}."
            )
            raise TypeError(msg)
        super().__init__(**kwargs)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @property
    def is_single_input(self) -> bool:
        """Whether the tool only accepts a single input."""
        keys = {k for k in self.args if k != "kwargs"}
        return len(keys) == 1

    @property
    def args(self) -> dict:
        return self.get_input_schema().model_json_schema()["properties"]

    @property
    def tool_call_schema(self) -> type[BaseModel]:
        full_schema = self.get_input_schema()
        fields = []
        for name, type_ in get_all_basemodel_annotations(full_schema).items():
            if not _is_injected_arg_type(type_):
                fields.append(name)
        return _create_subset_model(
            self.name, full_schema, fields, fn_description=self.description
        )

    # --- Runnable ---

    def get_input_schema(
        self, config: Optional[RunnableConfig] = None
    ) -> type[BaseModel]:
        """The tool's input schema.

        Args:
            config: The configuration for the tool.

        Returns:
            The input schema for the tool.
        """
        if self.args_schema is not None:
            return self.args_schema
        else:
            return create_schema_from_function(self.name, self._run)

    def invoke(
        self,
        input: Union[str, dict, ToolCall],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Any:
        tool_input, kwargs = _prep_run_args(input, config, **kwargs)
        return self.run(tool_input, **kwargs)

    async def ainvoke(
        self,
        input: Union[str, dict, ToolCall],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Any:
        tool_input, kwargs = _prep_run_args(input, config, **kwargs)
        return await self.arun(tool_input, **kwargs)

    # --- Tool ---

    def _parse_input(self, tool_input: Union[str, dict]) -> Union[str, dict[str, Any]]:
        """Convert tool input to a pydantic model.

        Args:
            tool_input: The input to the tool.
        """
        input_args = self.args_schema
        if isinstance(tool_input, str):
            if input_args is not None:
                key_ = next(iter(get_fields(input_args).keys()))
                if hasattr(input_args, "model_validate"):
                    input_args.model_validate({key_: tool_input})
                else:
                    input_args.parse_obj({key_: tool_input})
            return tool_input
        else:
            if input_args is not None:
                if issubclass(input_args, BaseModel):
                    result = input_args.model_validate(tool_input)
                    result_dict = result.model_dump()
                elif issubclass(input_args, BaseModelV1):
                    result = input_args.parse_obj(tool_input)
                    result_dict = result.dict()
                else:
                    msg = (
                        "args_schema must be a Pydantic BaseModel, "
                        f"got {self.args_schema}"
                    )
                    raise NotImplementedError(msg)
                return {
                    k: getattr(result, k)
                    for k, v in result_dict.items()
                    if k in tool_input
                }
            return tool_input

    @model_validator(mode="before")
    @classmethod
    def raise_deprecation(cls, values: dict) -> Any:
        """Raise deprecation warning if callback_manager is used.

        Args:
            values: The values to validate.

        Returns:
            The validated values.
        """
        if values.get("callback_manager") is not None:
            warnings.warn(
                "callback_manager is deprecated. Please use callbacks instead.",
                DeprecationWarning,
                stacklevel=6,
            )
            values["callbacks"] = values.pop("callback_manager", None)
        return values

    @abstractmethod
    def _run(self, *args: Any, **kwargs: Any) -> Any:
        """Use the tool.

        Add run_manager: Optional[CallbackManagerForToolRun] = None
        to child implementations to enable tracing.
        """

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        """Use the tool asynchronously.

        Add run_manager: Optional[AsyncCallbackManagerForToolRun] = None
        to child implementations to enable tracing.
        """
        if kwargs.get("run_manager") and signature(self._run).parameters.get(
            "run_manager"
        ):
            kwargs["run_manager"] = kwargs["run_manager"].get_sync()
        return await run_in_executor(None, self._run, *args, **kwargs)

    def _to_args_and_kwargs(self, tool_input: Union[str, dict]) -> tuple[tuple, dict]:
        tool_input = self._parse_input(tool_input)
        # For backwards compatibility, if run_input is a string,
        # pass as a positional argument.
        if isinstance(tool_input, str):
            return (tool_input,), {}
        else:
            return (), tool_input

    def run(
        self,
        tool_input: Union[str, dict[str, Any]],
        verbose: Optional[bool] = None,
        start_color: Optional[str] = "green",
        color: Optional[str] = "green",
        callbacks: Callbacks = None,
        *,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        run_name: Optional[str] = None,
        run_id: Optional[uuid.UUID] = None,
        config: Optional[RunnableConfig] = None,
        tool_call_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Run the tool.

        Args:
            tool_input: The input to the tool.
            verbose: Whether to log the tool's progress. Defaults to None.
            start_color: The color to use when starting the tool. Defaults to 'green'.
            color: The color to use when ending the tool. Defaults to 'green'.
            callbacks: Callbacks to be called during tool execution. Defaults to None.
            tags: Optional list of tags associated with the tool. Defaults to None.
            metadata: Optional metadata associated with the tool. Defaults to None.
            run_name: The name of the run. Defaults to None.
            run_id: The id of the run. Defaults to None.
            config: The configuration for the tool. Defaults to None.
            tool_call_id: The id of the tool call. Defaults to None.
            kwargs: Additional arguments to pass to the tool

        Returns:
            The output of the tool.

        Raises:
            ToolException: If an error occurs during tool execution.
        """
        callback_manager = CallbackManager.configure(
            callbacks,
            self.callbacks,
            self.verbose or bool(verbose),
            tags,
            self.tags,
            metadata,
            self.metadata,
        )

        run_manager = callback_manager.on_tool_start(
            {"name": self.name, "description": self.description},
            tool_input if isinstance(tool_input, str) else str(tool_input),
            color=start_color,
            name=run_name,
            run_id=run_id,
            # Inputs by definition should always be dicts.
            # For now, it's unclear whether this assumption is ever violated,
            # but if it is we will send a `None` value to the callback instead
            # TODO: will need to address issue via a patch.
            inputs=tool_input if isinstance(tool_input, dict) else None,
            **kwargs,
        )

        content = None
        artifact = None
        error_to_raise: Union[Exception, KeyboardInterrupt, None] = None
        try:
            child_config = patch_config(config, callbacks=run_manager.get_child())
            context = copy_context()
            context.run(_set_config_context, child_config)
            tool_args, tool_kwargs = self._to_args_and_kwargs(tool_input)
            if signature(self._run).parameters.get("run_manager"):
                tool_kwargs["run_manager"] = run_manager

            if config_param := _get_runnable_config_param(self._run):
                tool_kwargs[config_param] = config
            response = context.run(self._run, *tool_args, **tool_kwargs)
            if self.response_format == "content_and_artifact":
                if not isinstance(response, tuple) or len(response) != 2:
                    msg = (
                        "Since response_format='content_and_artifact' "
                        "a two-tuple of the message content and raw tool output is "
                        f"expected. Instead generated response of type: "
                        f"{type(response)}."
                    )
                    raise ValueError(msg)
                content, artifact = response
            else:
                content = response
            status = "success"
        except (ValidationError, ValidationErrorV1) as e:
            if not self.handle_validation_error:
                error_to_raise = e
            else:
                content = _handle_validation_error(e, flag=self.handle_validation_error)
            status = "error"
        except ToolException as e:
            if not self.handle_tool_error:
                error_to_raise = e
            else:
                content = _handle_tool_error(e, flag=self.handle_tool_error)
            status = "error"
        except (Exception, KeyboardInterrupt) as e:
            error_to_raise = e
            status = "error"

        if error_to_raise:
            run_manager.on_tool_error(error_to_raise)
            raise error_to_raise
        output = _format_output(content, artifact, tool_call_id, self.name, status)
        run_manager.on_tool_end(output, color=color, name=self.name, **kwargs)
        return output

    async def arun(
        self,
        tool_input: Union[str, dict],
        verbose: Optional[bool] = None,
        start_color: Optional[str] = "green",
        color: Optional[str] = "green",
        callbacks: Callbacks = None,
        *,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        run_name: Optional[str] = None,
        run_id: Optional[uuid.UUID] = None,
        config: Optional[RunnableConfig] = None,
        tool_call_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Run the tool asynchronously.

        Args:
            tool_input: The input to the tool.
            verbose: Whether to log the tool's progress. Defaults to None.
            start_color: The color to use when starting the tool. Defaults to 'green'.
            color: The color to use when ending the tool. Defaults to 'green'.
            callbacks: Callbacks to be called during tool execution. Defaults to None.
            tags: Optional list of tags associated with the tool. Defaults to None.
            metadata: Optional metadata associated with the tool. Defaults to None.
            run_name: The name of the run. Defaults to None.
            run_id: The id of the run. Defaults to None.
            config: The configuration for the tool. Defaults to None.
            tool_call_id: The id of the tool call. Defaults to None.
            kwargs: Additional arguments to pass to the tool

        Returns:
            The output of the tool.

        Raises:
            ToolException: If an error occurs during tool execution.
        """
        callback_manager = AsyncCallbackManager.configure(
            callbacks,
            self.callbacks,
            self.verbose or bool(verbose),
            tags,
            self.tags,
            metadata,
            self.metadata,
        )
        run_manager = await callback_manager.on_tool_start(
            {"name": self.name, "description": self.description},
            tool_input if isinstance(tool_input, str) else str(tool_input),
            color=start_color,
            name=run_name,
            run_id=run_id,
            # Inputs by definition should always be dicts.
            # For now, it's unclear whether this assumption is ever violated,
            # but if it is we will send a `None` value to the callback instead
            # TODO: will need to address issue via a patch.
            inputs=tool_input if isinstance(tool_input, dict) else None,
            **kwargs,
        )
        content = None
        artifact = None
        error_to_raise: Optional[Union[Exception, KeyboardInterrupt]] = None
        try:
            tool_args, tool_kwargs = self._to_args_and_kwargs(tool_input)
            child_config = patch_config(config, callbacks=run_manager.get_child())
            context = copy_context()
            context.run(_set_config_context, child_config)
            func_to_check = (
                self._run if self.__class__._arun is BaseTool._arun else self._arun
            )
            if signature(func_to_check).parameters.get("run_manager"):
                tool_kwargs["run_manager"] = run_manager
            if config_param := _get_runnable_config_param(func_to_check):
                tool_kwargs[config_param] = config

            coro = context.run(self._arun, *tool_args, **tool_kwargs)
            if asyncio_accepts_context():
                response = await asyncio.create_task(coro, context=context)  # type: ignore
            else:
                response = await coro
            if self.response_format == "content_and_artifact":
                if not isinstance(response, tuple) or len(response) != 2:
                    msg = (
                        "Since response_format='content_and_artifact' "
                        "a two-tuple of the message content and raw tool output is "
                        f"expected. Instead generated response of type: "
                        f"{type(response)}."
                    )
                    raise ValueError(msg)
                content, artifact = response
            else:
                content = response
            status = "success"
        except ValidationError as e:
            if not self.handle_validation_error:
                error_to_raise = e
            else:
                content = _handle_validation_error(e, flag=self.handle_validation_error)
            status = "error"
        except ToolException as e:
            if not self.handle_tool_error:
                error_to_raise = e
            else:
                content = _handle_tool_error(e, flag=self.handle_tool_error)
            status = "error"
        except (Exception, KeyboardInterrupt) as e:
            error_to_raise = e
            status = "error"

        if error_to_raise:
            await run_manager.on_tool_error(error_to_raise)
            raise error_to_raise

        output = _format_output(content, artifact, tool_call_id, self.name, status)
        await run_manager.on_tool_end(output, color=color, name=self.name, **kwargs)
        return output

    @deprecated("0.1.47", alternative="invoke", removal="1.0")
    def __call__(self, tool_input: str, callbacks: Callbacks = None) -> str:
        """Make tool callable."""
        return self.run(tool_input, callbacks=callbacks)


def _is_tool_call(x: Any) -> bool:
    return isinstance(x, dict) and x.get("type") == "tool_call"


def _handle_validation_error(
    e: Union[ValidationError, ValidationErrorV1],
    *,
    flag: Union[
        Literal[True], str, Callable[[Union[ValidationError, ValidationErrorV1]], str]
    ],
) -> str:
    if isinstance(flag, bool):
        content = "Tool input validation error"
    elif isinstance(flag, str):
        content = flag
    elif callable(flag):
        content = flag(e)
    else:
        msg = (
            f"Got unexpected type of `handle_validation_error`. Expected bool, "
            f"str or callable. Received: {flag}"
        )
        raise ValueError(msg)
    return content


def _handle_tool_error(
    e: ToolException,
    *,
    flag: Optional[Union[Literal[True], str, Callable[[ToolException], str]]],
) -> str:
    if isinstance(flag, bool):
        content = e.args[0] if e.args else "Tool execution error"
    elif isinstance(flag, str):
        content = flag
    elif callable(flag):
        content = flag(e)
    else:
        msg = (
            f"Got unexpected type of `handle_tool_error`. Expected bool, str "
            f"or callable. Received: {flag}"
        )
        raise ValueError(msg)
    return content


def _prep_run_args(
    input: Union[str, dict, ToolCall],
    config: Optional[RunnableConfig],
    **kwargs: Any,
) -> tuple[Union[str, dict], dict]:
    config = ensure_config(config)
    if _is_tool_call(input):
        tool_call_id: Optional[str] = cast(ToolCall, input)["id"]
        tool_input: Union[str, dict] = cast(ToolCall, input)["args"].copy()
    else:
        tool_call_id = None
        tool_input = cast(Union[str, dict], input)
    return (
        tool_input,
        dict(
            callbacks=config.get("callbacks"),
            tags=config.get("tags"),
            metadata=config.get("metadata"),
            run_name=config.get("run_name"),
            run_id=config.pop("run_id", None),
            config=config,
            tool_call_id=tool_call_id,
            **kwargs,
        ),
    )


def _format_output(
    content: Any, artifact: Any, tool_call_id: Optional[str], name: str, status: str
) -> Union[ToolMessage, Any]:
    if tool_call_id:
        if not _is_message_content_type(content):
            content = _stringify(content)
        return ToolMessage(
            content,
            artifact=artifact,
            tool_call_id=tool_call_id,
            name=name,
            status=status,
        )
    else:
        return content


def _is_message_content_type(obj: Any) -> bool:
    """Check for OpenAI or Anthropic format tool message content."""
    return (
        isinstance(obj, str)
        or isinstance(obj, list)
        and all(_is_message_content_block(e) for e in obj)
    )


def _is_message_content_block(obj: Any) -> bool:
    """Check for OpenAI or Anthropic format tool message content blocks."""
    if isinstance(obj, str):
        return True
    elif isinstance(obj, dict):
        return obj.get("type", None) in ("text", "image_url", "image", "json")
    else:
        return False


def _stringify(content: Any) -> str:
    try:
        return json.dumps(content, ensure_ascii=False)
    except Exception:
        return str(content)


def _get_type_hints(func: Callable) -> Optional[dict[str, type]]:
    if isinstance(func, functools.partial):
        func = func.func
    try:
        return get_type_hints(func)
    except Exception:
        return None


def _get_runnable_config_param(func: Callable) -> Optional[str]:
    type_hints = _get_type_hints(func)
    if not type_hints:
        return None
    for name, type_ in type_hints.items():
        if type_ is RunnableConfig:
            return name
    return None


class InjectedToolArg:
    """Annotation for a Tool arg that is **not** meant to be generated by a model."""


def _is_injected_arg_type(type_: type) -> bool:
    return any(
        isinstance(arg, InjectedToolArg)
        or (isinstance(arg, type) and issubclass(arg, InjectedToolArg))
        for arg in get_args(type_)[1:]
    )


def get_all_basemodel_annotations(
    cls: Union[TypeBaseModel, Any], *, default_to_bound: bool = True
) -> dict[str, type]:
    # cls has no subscript: cls = FooBar
    if isinstance(cls, type):
        annotations: dict[str, type] = {}
        for name, param in inspect.signature(cls).parameters.items():
            # Exclude hidden init args added by pydantic Config. For example if
            # BaseModel(extra="allow") then "extra_data" will part of init sig.
            if (
                fields := getattr(cls, "model_fields", {})  # pydantic v2+
                or getattr(cls, "__fields__", {})  # pydantic v1
            ) and name not in fields:
                continue
            annotations[name] = param.annotation
        orig_bases: tuple = getattr(cls, "__orig_bases__", ())
    # cls has subscript: cls = FooBar[int]
    else:
        annotations = get_all_basemodel_annotations(
            get_origin(cls), default_to_bound=False
        )
        orig_bases = (cls,)

    # Pydantic v2 automatically resolves inherited generics, Pydantic v1 does not.
    if not (isinstance(cls, type) and is_pydantic_v2_subclass(cls)):
        # if cls = FooBar inherits from Baz[str], orig_bases will contain Baz[str]
        # if cls = FooBar inherits from Baz, orig_bases will contain Baz
        # if cls = FooBar[int], orig_bases will contain FooBar[int]
        for parent in orig_bases:
            # if class = FooBar inherits from Baz, parent = Baz
            if isinstance(parent, type) and is_pydantic_v1_subclass(parent):
                annotations.update(
                    get_all_basemodel_annotations(parent, default_to_bound=False)
                )
                continue

            parent_origin = get_origin(parent)

            # if class = FooBar inherits from non-pydantic class
            if not parent_origin:
                continue

            # if class = FooBar inherits from Baz[str]:
            # parent = Baz[str],
            # parent_origin = Baz,
            # generic_type_vars = (type vars in Baz)
            # generic_map = {type var in Baz: str}
            generic_type_vars: tuple = getattr(parent_origin, "__parameters__", ())
            generic_map = dict(zip(generic_type_vars, get_args(parent)))
            for field in getattr(parent_origin, "__annotations__", {}):
                annotations[field] = _replace_type_vars(
                    annotations[field], generic_map, default_to_bound
                )

    return {
        k: _replace_type_vars(v, default_to_bound=default_to_bound)
        for k, v in annotations.items()
    }


def _replace_type_vars(
    type_: type,
    generic_map: Optional[dict[TypeVar, type]] = None,
    default_to_bound: bool = True,
) -> type:
    generic_map = generic_map or {}
    if isinstance(type_, TypeVar):
        if type_ in generic_map:
            return generic_map[type_]
        elif default_to_bound:
            return type_.__bound__ or Any
        else:
            return type_
    elif (origin := get_origin(type_)) and (args := get_args(type_)):
        new_args = tuple(
            _replace_type_vars(arg, generic_map, default_to_bound) for arg in args
        )
        return _py_38_safe_origin(origin)[new_args]  # type: ignore[index]
    else:
        return type_


class BaseToolkit(BaseModel, ABC):
    """Base Toolkit representing a collection of related tools."""

    @abstractmethod
    def get_tools(self) -> list[BaseTool]:
        """Get the tools in the toolkit."""
