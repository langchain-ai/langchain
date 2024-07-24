from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Mapping,
    Optional,
    Type,
    TypeVar,
    Union,
)

import yaml

from langchain_core.output_parsers.base import BaseOutputParser
from langchain_core.prompt_values import (
    ChatPromptValueConcrete,
    PromptValue,
    StringPromptValue,
)
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator
from langchain_core.runnables import RunnableConfig, RunnableSerializable
from langchain_core.runnables.config import ensure_config
from langchain_core.runnables.utils import create_model

if TYPE_CHECKING:
    from langchain_core.documents import Document


FormatOutputType = TypeVar("FormatOutputType")


class BasePromptTemplate(
    RunnableSerializable[Dict, PromptValue], Generic[FormatOutputType], ABC
):
    """Base class for all prompt templates, returning a prompt."""

    input_variables: List[str]
    """A list of the names of the variables whose values are required as inputs to the 
    prompt."""
    optional_variables: List[str] = Field(default=[])
    """A list of the names of the variables that are optional in the prompt."""
    input_types: Dict[str, Any] = Field(default_factory=dict, exclude=True)
    """A dictionary of the types of the variables the prompt template expects.
    If not provided, all variables are assumed to be strings."""
    output_parser: Optional[BaseOutputParser] = None
    """How to parse the output of calling an LLM on this formatted prompt."""
    partial_variables: Mapping[str, Any] = Field(default_factory=dict)
    """A dictionary of the partial variables the prompt template carries.
    
    Partial variables populate the template so that you don't need to
    pass them in every time you call the prompt."""
    metadata: Optional[Dict[str, Any]] = None
    """Metadata to be used for tracing."""
    tags: Optional[List[str]] = None
    """Tags to be used for tracing."""

    @root_validator(pre=False, skip_on_failure=True)
    def validate_variable_names(cls, values: Dict) -> Dict:
        """Validate variable names do not include restricted names."""
        if "stop" in values["input_variables"]:
            raise ValueError(
                "Cannot have an input variable named 'stop', as it is used internally,"
                " please rename."
            )
        if "stop" in values["partial_variables"]:
            raise ValueError(
                "Cannot have an partial variable named 'stop', as it is used "
                "internally, please rename."
            )

        overall = set(values["input_variables"]).intersection(
            values["partial_variables"]
        )
        if overall:
            raise ValueError(
                f"Found overlapping input and partial variables: {overall}"
            )
        return values

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object.
        Returns ["langchain", "schema", "prompt_template"]."""
        return ["langchain", "schema", "prompt_template"]

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this class is serializable.
        Returns True."""
        return True

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    @property
    def OutputType(self) -> Any:
        """Return the output type of the prompt."""
        return Union[StringPromptValue, ChatPromptValueConcrete]

    def get_input_schema(
        self, config: Optional[RunnableConfig] = None
    ) -> Type[BaseModel]:
        """Get the input schema for the prompt.

        Args:
            config: RunnableConfig, configuration for the prompt.

        Returns:
            Type[BaseModel]: The input schema for the prompt.
        """
        # This is correct, but pydantic typings/mypy don't think so.
        required_input_variables = {
            k: (self.input_types.get(k, str), ...) for k in self.input_variables
        }
        optional_input_variables = {
            k: (self.input_types.get(k, str), None) for k in self.optional_variables
        }
        return create_model(
            "PromptInput", **{**required_input_variables, **optional_input_variables}
        )

    def _validate_input(self, inner_input: Dict) -> Dict:
        if not isinstance(inner_input, dict):
            if len(self.input_variables) == 1:
                var_name = self.input_variables[0]
                inner_input = {var_name: inner_input}

            else:
                raise TypeError(
                    f"Expected mapping type as input to {self.__class__.__name__}. "
                    f"Received {type(inner_input)}."
                )
        missing = set(self.input_variables).difference(inner_input)
        if missing:
            raise KeyError(
                f"Input to {self.__class__.__name__} is missing variables {missing}. "
                f" Expected: {self.input_variables}"
                f" Received: {list(inner_input.keys())}"
            )
        return inner_input

    def _format_prompt_with_error_handling(self, inner_input: Dict) -> PromptValue:
        _inner_input = self._validate_input(inner_input)
        return self.format_prompt(**_inner_input)

    async def _aformat_prompt_with_error_handling(
        self, inner_input: Dict
    ) -> PromptValue:
        _inner_input = self._validate_input(inner_input)
        return await self.aformat_prompt(**_inner_input)

    def invoke(
        self, input: Dict, config: Optional[RunnableConfig] = None
    ) -> PromptValue:
        """Invoke the prompt.

        Args:
            input: Dict, input to the prompt.
            config: RunnableConfig, configuration for the prompt.

        Returns:
            PromptValue: The output of the prompt.
        """
        config = ensure_config(config)
        if self.metadata:
            config["metadata"] = {**config["metadata"], **self.metadata}
        if self.tags:
            config["tags"] = config["tags"] + self.tags
        return self._call_with_config(
            self._format_prompt_with_error_handling,
            input,
            config,
            run_type="prompt",
        )

    async def ainvoke(
        self, input: Dict, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> PromptValue:
        """Async invoke the prompt.

        Args:
            input: Dict, input to the prompt.
            config: RunnableConfig, configuration for the prompt.

        Returns:
            PromptValue: The output of the prompt.
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
        )

    @abstractmethod
    def format_prompt(self, **kwargs: Any) -> PromptValue:
        """Create Prompt Value.

        Args:
            kwargs: Any arguments to be passed to the prompt template.

        Returns:
            PromptValue: The output of the prompt.
        """

    async def aformat_prompt(self, **kwargs: Any) -> PromptValue:
        """Async create Prompt Value.

        Args:
            kwargs: Any arguments to be passed to the prompt template.

        Returns:
            PromptValue: The output of the prompt.
        """
        return self.format_prompt(**kwargs)

    def partial(self, **kwargs: Union[str, Callable[[], str]]) -> BasePromptTemplate:
        """Return a partial of the prompt template.

        Args:
            kwargs: Union[str, Callable[[], str], partial variables to set.

        Returns:
            BasePromptTemplate: A partial of the prompt template.
        """
        prompt_dict = self.__dict__.copy()
        prompt_dict["input_variables"] = list(
            set(self.input_variables).difference(kwargs)
        )
        prompt_dict["partial_variables"] = {**self.partial_variables, **kwargs}
        return type(self)(**prompt_dict)

    def _merge_partial_and_user_variables(self, **kwargs: Any) -> Dict[str, Any]:
        # Get partial params:
        partial_kwargs = {
            k: v if not callable(v) else v() for k, v in self.partial_variables.items()
        }
        return {**partial_kwargs, **kwargs}

    @abstractmethod
    def format(self, **kwargs: Any) -> FormatOutputType:
        """Format the prompt with the inputs.

        Args:
            kwargs: Any arguments to be passed to the prompt template.

        Returns:
            A formatted string.

        Example:

        .. code-block:: python

            prompt.format(variable1="foo")
        """

    async def aformat(self, **kwargs: Any) -> FormatOutputType:
        """Async format the prompt with the inputs.

        Args:
            kwargs: Any arguments to be passed to the prompt template.

        Returns:
            A formatted string.

        Example:

        .. code-block:: python

            await prompt.aformat(variable1="foo")
        """
        return self.format(**kwargs)

    @property
    def _prompt_type(self) -> str:
        """Return the prompt type key."""
        raise NotImplementedError

    def dict(self, **kwargs: Any) -> Dict:
        """Return dictionary representation of prompt.

        Args:
            kwargs: Any additional arguments to pass to the dictionary.

        Returns:
            Dict: Dictionary representation of the prompt.

        Raises:
            NotImplementedError: If the prompt type is not implemented.
        """
        prompt_dict = super().dict(**kwargs)
        try:
            prompt_dict["_type"] = self._prompt_type
        except NotImplementedError:
            pass
        return prompt_dict

    def save(self, file_path: Union[Path, str]) -> None:
        """Save the prompt.

        Args:
            file_path: Path to directory to save prompt to.

        Raises:
            ValueError: If the prompt has partial variables.
            ValueError: If the file path is not json or yaml.
            NotImplementedError: If the prompt type is not implemented.

        Example:
        .. code-block:: python

            prompt.save(file_path="path/prompt.yaml")
        """
        if self.partial_variables:
            raise ValueError("Cannot save prompt with partial variables.")

        # Fetch dictionary to save
        prompt_dict = self.dict()
        if "_type" not in prompt_dict:
            raise NotImplementedError(f"Prompt {self} does not support saving.")

        # Convert file to Path object.
        if isinstance(file_path, str):
            save_path = Path(file_path)
        else:
            save_path = file_path

        directory_path = save_path.parent
        directory_path.mkdir(parents=True, exist_ok=True)

        if save_path.suffix == ".json":
            with open(file_path, "w") as f:
                json.dump(prompt_dict, f, indent=4)
        elif save_path.suffix.endswith((".yaml", ".yml")):
            with open(file_path, "w") as f:
                yaml.dump(prompt_dict, f, default_flow_style=False)
        else:
            raise ValueError(f"{save_path} must be json or yaml")


def _get_document_info(doc: Document, prompt: BasePromptTemplate[str]) -> Dict:
    base_info = {"page_content": doc.page_content, **doc.metadata}
    missing_metadata = set(prompt.input_variables).difference(base_info)
    if len(missing_metadata) > 0:
        required_metadata = [
            iv for iv in prompt.input_variables if iv != "page_content"
        ]
        raise ValueError(
            f"Document prompt requires documents to have metadata variables: "
            f"{required_metadata}. Received document with missing metadata: "
            f"{list(missing_metadata)}."
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
        .. code-block:: python

            from langchain_core.documents import Document
            from langchain_core.prompts import PromptTemplate

            doc = Document(page_content="This is a joke", metadata={"page": "1"})
            prompt = PromptTemplate.from_template("Page {page}: {page_content}")
            format_document(doc, prompt)
            >>> "Page 1: This is a joke"
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
