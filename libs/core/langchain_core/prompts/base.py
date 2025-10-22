"""Base class for prompt templates."""

from __future__ import annotations

import contextlib
import json
import typing
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from functools import cached_property
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    TypeVar,
)

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self, override

from langchain_core.exceptions import ErrorCode, create_message
from langchain_core.load import dumpd
from langchain_core.output_parsers.base import BaseOutputParser
from langchain_core.prompt_values import (
    ChatPromptValueConcrete,
    PromptValue,
    StringPromptValue,
)
from langchain_core.runnables import RunnableConfig, RunnableSerializable
from langchain_core.runnables.config import ensure_config
from langchain_core.utils.pydantic import create_model_v2

if TYPE_CHECKING:
    from langchain_core.documents import Document


FormatOutputType = TypeVar("FormatOutputType")


class BasePromptTemplate(
    RunnableSerializable[dict, PromptValue], ABC, Generic[FormatOutputType]
):
    """Base class for all prompt templates, returning a prompt."""

    input_variables: list[str]
    """A list of the names of the variables whose values are required as inputs to the
    prompt."""
    optional_variables: list[str] = Field(default=[])
    """optional_variables: A list of the names of the variables for placeholder
       or MessagePlaceholder that are optional. These variables are auto inferred
       from the prompt and user need not provide them."""
    input_types: typing.Dict[str, Any] = Field(default_factory=dict, exclude=True)  # noqa: UP006
    """A dictionary of the types of the variables the prompt template expects.
    If not provided, all variables are assumed to be strings."""
    output_parser: BaseOutputParser | None = None
    """How to parse the output of calling an LLM on this formatted prompt."""
    partial_variables: Mapping[str, Any] = Field(default_factory=dict)
    """A dictionary of the partial variables the prompt template carries.

    Partial variables populate the template so that you don't need to
    pass them in every time you call the prompt."""
    metadata: typing.Dict[str, Any] | None = None  # noqa: UP006
    """Metadata to be used for tracing."""
    tags: list[str] | None = None
    """Tags to be used for tracing."""

    @model_validator(mode="after")
    def validate_variable_names(self) -> Self:
        """Validate variable names do not include restricted names."""
        if "stop" in self.input_variables:
            msg = (
                "Cannot have an input variable named 'stop', as it is used internally,"
                " please rename."
            )
            raise ValueError(
                create_message(message=msg, error_code=ErrorCode.INVALID_PROMPT_INPUT)
            )
        if "stop" in self.partial_variables:
            msg = (
                "Cannot have an partial variable named 'stop', as it is used "
                "internally, please rename."
            )
            raise ValueError(
                create_message(message=msg, error_code=ErrorCode.INVALID_PROMPT_INPUT)
            )

        overall = set(self.input_variables).intersection(self.partial_variables)
        if overall:
            msg = f"Found overlapping input and partial variables: {overall}"
            raise ValueError(
                create_message(message=msg, error_code=ErrorCode.INVALID_PROMPT_INPUT)
            )
        return self

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the LangChain object.

        Returns:
            `["langchain", "schema", "prompt_template"]`
        """
        return ["langchain", "schema", "prompt_template"]

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return True as this class is serializable."""
        return True

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @cached_property
    def _serialized(self) -> dict[str, Any]:
        return dumpd(self)

    @property
    @override
    def OutputType(self) -> Any:
        """Return the output type of the prompt."""
        return StringPromptValue | ChatPromptValueConcrete

    @override
    def get_input_schema(self, config: RunnableConfig | None = None) -> type[BaseModel]:
        """Get the input schema for the prompt.

        Args:
            config: configuration for the prompt.

        Returns:
            The input schema for the prompt.
        """
        # This is correct, but pydantic typings/mypy don't think so.
        required_input_variables = {
            k: (self.input_types.get(k, str), ...) for k in self.input_variables
        }
        optional_input_variables = {
            k: (self.input_types.get(k, str), None) for k in self.optional_variables
        }
        return create_model_v2(
            "PromptInput",
            field_definitions={**required_input_variables, **optional_input_variables},
        )

    def _validate_input(self, inner_input: Any) -> dict:
        if not isinstance(inner_input, dict):
            if len(self.input_variables) == 1:
                var_name = self.input_variables[0]
                inner_input = {var_name: inner_input}

            else:
                msg = (
                    f"Expected mapping type as input to {self.__class__.__name__}. "
                    f"Received {type(inner_input)}."
                )
                raise TypeError(
                    create_message(
                        message=msg, error_code=ErrorCode.INVALID_PROMPT_INPUT
                    )
                )
        missing = set(self.input_variables).difference(inner_input)
        if missing:
            msg = (
                f"Input to {self.__class__.__name__} is missing variables {missing}. "
                f" Expected: {self.input_variables}"
                f" Received: {list(inner_input.keys())}"
            )
            example_key = missing.pop()
            msg += (
                f"\nNote: if you intended {{{example_key}}} to be part of the string"
                " and not a variable, please escape it with double curly braces like: "
                f"'{{{{{example_key}}}}}'."
            )
            raise KeyError(
                create_message(message=msg, error_code=ErrorCode.INVALID_PROMPT_INPUT)
            )
        return inner_input

    def _format_prompt_with_error_handling(self, inner_input: dict) -> PromptValue:
        inner_input_ = self._validate_input(inner_input)
        return self.format_prompt(**inner_input_)

    async def _aformat_prompt_with_error_handling(
        self, inner_input: dict
    ) -> PromptValue:
        inner_input_ = self._validate_input(inner_input)
        return await self.aformat_prompt(**inner_input_)

    @override
    def invoke(
        self, input: dict, config: RunnableConfig | None = None, **kwargs: Any
    ) -> PromptValue:
        """Invoke the prompt.

        Args:
            input: Dict, input to the prompt.
            config: RunnableConfig, configuration for the prompt.

        Returns:
            The output of the prompt.
        """
        config = ensure_config(config)
        if self.metadata:
            config["metadata"] = {**config["metadata"], **self.metadata}
        if self.tags:
            config["tags"] += self.tags
        return self._call_with_config(
            self._format_prompt_with_error_handling,
            input,
            config,
            run_type="prompt",
            serialized=self._serialized,
        )

    @override
    async def ainvoke(
        self, input: dict, config: RunnableConfig | None = None, **kwargs: Any
    ) -> PromptValue:
        """Async invoke the prompt.

        Args:
            input: Dict, input to the prompt.
            config: RunnableConfig, configuration for the prompt.

        Returns:
            The output of the prompt.
        """
        config = ensure_config(config)
        if self.metadata:
            config["metadata"].update(self.metadata)
        if self.tags:
            config["tags"].extend(self.tags)
        return await self._acall_with_config(
            self._aformat_prompt_with_error_handling,
            input,
            config,
            run_type="prompt",
            serialized=self._serialized,
        )

    @abstractmethod
    def format_prompt(self, **kwargs: Any) -> PromptValue:
        """Create Prompt Value.

        Args:
            **kwargs: Any arguments to be passed to the prompt template.

        Returns:
            The output of the prompt.
        """

    async def aformat_prompt(self, **kwargs: Any) -> PromptValue:
        """Async create Prompt Value.

        Args:
            **kwargs: Any arguments to be passed to the prompt template.

        Returns:
            The output of the prompt.
        """
        return self.format_prompt(**kwargs)

    def partial(self, **kwargs: str | Callable[[], str]) -> BasePromptTemplate:
        """Return a partial of the prompt template.

        Args:
            **kwargs: partial variables to set.

        Returns:
            A partial of the prompt template.
        """
        prompt_dict = self.__dict__.copy()
        prompt_dict["input_variables"] = list(
            set(self.input_variables).difference(kwargs)
        )
        prompt_dict["partial_variables"] = {**self.partial_variables, **kwargs}
        return type(self)(**prompt_dict)

    def _merge_partial_and_user_variables(self, **kwargs: Any) -> dict[str, Any]:
        # Get partial params:
        partial_kwargs = {
            k: v if not callable(v) else v() for k, v in self.partial_variables.items()
        }
        return {**partial_kwargs, **kwargs}

    @abstractmethod
    def format(self, **kwargs: Any) -> FormatOutputType:
        """Format the prompt with the inputs.

        Args:
            **kwargs: Any arguments to be passed to the prompt template.

        Returns:
            A formatted string.

        Example:
        ```python
        prompt.format(variable1="foo")
        ```
        """

    async def aformat(self, **kwargs: Any) -> FormatOutputType:
        """Async format the prompt with the inputs.

        Args:
            **kwargs: Any arguments to be passed to the prompt template.

        Returns:
            A formatted string.

        Example:
        ```python
        await prompt.aformat(variable1="foo")
        ```
        """
        return self.format(**kwargs)

    @property
    def _prompt_type(self) -> str:
        """Return the prompt type key."""
        raise NotImplementedError

    def dict(self, **kwargs: Any) -> dict:
        """Return dictionary representation of prompt.

        Args:
            **kwargs: Any additional arguments to pass to the dictionary.

        Returns:
            Dictionary representation of the prompt.
        """
        prompt_dict = super().model_dump(**kwargs)
        with contextlib.suppress(NotImplementedError):
            prompt_dict["_type"] = self._prompt_type
        return prompt_dict

    def save(self, file_path: Path | str) -> None:
        """Save the prompt.

        Args:
            file_path: Path to directory to save prompt to.

        Raises:
            ValueError: If the prompt has partial variables.
            ValueError: If the file path is not json or yaml.
            NotImplementedError: If the prompt type is not implemented.

        Example:
        ```python
        prompt.save(file_path="path/prompt.yaml")
        ```
        """
        if self.partial_variables:
            msg = "Cannot save prompt with partial variables."
            raise ValueError(msg)

        # Fetch dictionary to save
        prompt_dict = self.dict()
        if "_type" not in prompt_dict:
            msg = f"Prompt {self} does not support saving."
            raise NotImplementedError(msg)

        # Convert file to Path object.
        save_path = Path(file_path)

        directory_path = save_path.parent
        directory_path.mkdir(parents=True, exist_ok=True)

        if save_path.suffix == ".json":
            with save_path.open("w", encoding="utf-8") as f:
                json.dump(prompt_dict, f, indent=4)
        elif save_path.suffix.endswith((".yaml", ".yml")):
            with save_path.open("w", encoding="utf-8") as f:
                yaml.dump(prompt_dict, f, default_flow_style=False)
        else:
            msg = f"{save_path} must be json or yaml"
            raise ValueError(msg)


def _get_document_info(doc: Document, prompt: BasePromptTemplate[str]) -> dict:
    base_info = {"page_content": doc.page_content, **doc.metadata}
    missing_metadata = set(prompt.input_variables).difference(base_info)
    if len(missing_metadata) > 0:
        required_metadata = [
            iv for iv in prompt.input_variables if iv != "page_content"
        ]
        msg = (
            f"Document prompt requires documents to have metadata variables: "
            f"{required_metadata}. Received document with missing metadata: "
            f"{list(missing_metadata)}."
        )
        raise ValueError(
            create_message(message=msg, error_code=ErrorCode.INVALID_PROMPT_INPUT)
        )
    return {k: base_info[k] for k in prompt.input_variables}


def format_document(doc: Document, prompt: BasePromptTemplate[str]) -> str:
    """Format a document into a string based on a prompt template.

    First, this pulls information from the document from two sources:

    1. page_content:
        This takes the information from the `document.page_content`
        and assigns it to a variable named `page_content`.
    2. metadata:
        This takes information from `document.metadata` and assigns
        it to variables of the same name.

    Those variables are then passed into the `prompt` to produce a formatted string.

    Args:
        doc: Document, the page_content and metadata will be used to create
            the final string.
        prompt: BasePromptTemplate, will be used to format the page_content
            and metadata into the final string.

    Returns:
        string of the document formatted.

    Example:
        ```python
        from langchain_core.documents import Document
        from langchain_core.prompts import PromptTemplate

        doc = Document(page_content="This is a joke", metadata={"page": "1"})
        prompt = PromptTemplate.from_template("Page {page}: {page_content}")
        format_document(doc, prompt)
        >>> "Page 1: This is a joke"

        ```
    """
    return prompt.format(**_get_document_info(doc, prompt))


async def aformat_document(doc: Document, prompt: BasePromptTemplate[str]) -> str:
    """Async format a document into a string based on a prompt template.

    First, this pulls information from the document from two sources:

    1. page_content:
        This takes the information from the `document.page_content`
        and assigns it to a variable named `page_content`.
    2. metadata:
        This takes information from `document.metadata` and assigns
        it to variables of the same name.

    Those variables are then passed into the `prompt` to produce a formatted string.

    Args:
        doc: Document, the page_content and metadata will be used to create
            the final string.
        prompt: BasePromptTemplate, will be used to format the page_content
            and metadata into the final string.

    Returns:
        string of the document formatted.
    """
    return await prompt.aformat(**_get_document_info(doc, prompt))
