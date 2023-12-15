from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Type,
    Union,
)

import yaml

from langchain_core.output_parsers.base import BaseOutputParser
from langchain_core.prompt_values import (
    ChatPromptValueConcrete,
    PromptValue,
    StringPromptValue,
)
from langchain_core.pydantic_v1 import BaseModel, Field, create_model, root_validator
from langchain_core.runnables import RunnableConfig, RunnableSerializable

if TYPE_CHECKING:
    from langchain_core.documents import Document


class BasePromptTemplate(RunnableSerializable[Dict, PromptValue], ABC):
    """Base class for all prompt templates, returning a prompt."""

    input_variables: List[str]
    """A list of the names of the variables the prompt template expects."""
    input_types: Dict[str, Any] = Field(default_factory=dict)
    """A dictionary of the types of the variables the prompt template expects.
    If not provided, all variables are assumed to be strings."""
    output_parser: Optional[BaseOutputParser] = None
    """How to parse the output of calling an LLM on this formatted prompt."""
    partial_variables: Mapping[str, Union[str, Callable[[], str]]] = Field(
        default_factory=dict
    )

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "schema", "prompt_template"]

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this class is serializable."""
        return True

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    @property
    def OutputType(self) -> Any:
        return Union[StringPromptValue, ChatPromptValueConcrete]

    def get_input_schema(
        self, config: Optional[RunnableConfig] = None
    ) -> Type[BaseModel]:
        # This is correct, but pydantic typings/mypy don't think so.
        return create_model(  # type: ignore[call-overload]
            "PromptInput",
            **{k: (self.input_types.get(k, str), None) for k in self.input_variables},
        )

    def _format_prompt_with_error_handling(self, inner_input: Dict) -> PromptValue:
        try:
            input_dict = {key: inner_input[key] for key in self.input_variables}
        except TypeError as e:
            raise TypeError(
                f"Expected mapping type as input to {self.__class__.__name__}. "
                f"Received {type(inner_input)}."
            ) from e
        except KeyError as e:
            raise KeyError(
                f"Input to {self.__class__.__name__} is missing variable {e}. "
                f" Expected: {self.input_variables}"
                f" Received: {list(inner_input.keys())}"
            ) from e
        return self.format_prompt(**input_dict)

    def invoke(
        self, input: Dict, config: Optional[RunnableConfig] = None
    ) -> PromptValue:
        return self._call_with_config(
            self._format_prompt_with_error_handling,
            input,
            config,
            run_type="prompt",
        )

    @abstractmethod
    def format_prompt(self, **kwargs: Any) -> PromptValue:
        """Create Chat Messages."""

    @root_validator()
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

    def partial(self, **kwargs: Union[str, Callable[[], str]]) -> BasePromptTemplate:
        """Return a partial of the prompt template."""
        prompt_dict = self.__dict__.copy()
        prompt_dict["input_variables"] = list(
            set(self.input_variables).difference(kwargs)
        )
        prompt_dict["partial_variables"] = {**self.partial_variables, **kwargs}
        return type(self)(**prompt_dict)

    def _merge_partial_and_user_variables(self, **kwargs: Any) -> Dict[str, Any]:
        # Get partial params:
        partial_kwargs = {
            k: v if isinstance(v, str) else v()
            for k, v in self.partial_variables.items()
        }
        return {**partial_kwargs, **kwargs}

    @abstractmethod
    def format(self, **kwargs: Any) -> str:
        """Format the prompt with the inputs.

        Args:
            kwargs: Any arguments to be passed to the prompt template.

        Returns:
            A formatted string.

        Example:

        .. code-block:: python

            prompt.format(variable1="foo")
        """

    @property
    def _prompt_type(self) -> str:
        """Return the prompt type key."""
        raise NotImplementedError

    def dict(self, **kwargs: Any) -> Dict:
        """Return dictionary representation of prompt."""
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
        elif save_path.suffix == ".yaml":
            with open(file_path, "w") as f:
                yaml.dump(prompt_dict, f, default_flow_style=False)
        else:
            raise ValueError(f"{save_path} must be json or yaml")


def format_document(doc: Document, prompt: BasePromptTemplate) -> str:
    """Format a document into a string based on a prompt template.

    First, this pulls information from the document from two sources:

    1. `page_content`:
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

            from langchain_core import Document
            from langchain_core.prompts import PromptTemplate

            doc = Document(page_content="This is a joke", metadata={"page": "1"})
            prompt = PromptTemplate.from_template("Page {page}: {page_content}")
            format_document(doc, prompt)
            >>> "Page 1: This is a joke"
    """
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
    document_info = {k: base_info[k] for k in prompt.input_variables}
    return prompt.format(**document_info)
