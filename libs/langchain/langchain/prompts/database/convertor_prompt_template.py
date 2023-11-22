"""
A "convertor"-based prompt template
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Union

from langchain_core.prompts.string import (
    DEFAULT_FORMATTER_MAPPING,
    StringPromptTemplate,
)

from langchain.pydantic_v1 import Extra, Field, root_validator

ConvertorType = Callable[[Dict[str, Any]], Dict[str, Any]]


class ConvertorPromptTemplate(StringPromptTemplate):
    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Is this class serializable?"""
        return False

    # generalization to accommodate arbitrary-type input variables
    partial_variables: Mapping[str, Union[Any, Callable[[], Any]]] = Field(
        default_factory=dict
    )

    template: str

    validate_template: bool = True

    input_variables: List[str]

    convertor: ConvertorType

    convertor_input_variables: List[str]

    convertor_output_variables: List[str]

    template_format: str = "f-string"

    class Config:
        extra = Extra.forbid
        arbitrary_types_allowed = True

    @root_validator(pre=False)
    def check_convertor_and_template(cls, values: Dict) -> Dict:
        # this is to ensure partialing knows what to do for e.g. ChatPromptTemplate
        values["input_variables"] = list(
            set(values["input_variables"]) | set(values["convertor_input_variables"])
        )
        return values

    def format(self, **kwargs: Any) -> str:
        kwargs = self._merge_partial_and_user_variables(**kwargs)

        convertor_kwargs = {
            k: v for k, v in kwargs.items() if k in self.convertor_input_variables
        }
        prompt_kwargs = {
            k: v for k, v in kwargs.items() if k not in self.convertor_input_variables
        }

        _converted = self.convertor(convertor_kwargs)
        # restrict to those which are featured in the prompt
        converted_kwargs = {k: _converted[k] for k in self.convertor_output_variables}

        full_kwargs = {**prompt_kwargs, **converted_kwargs}

        # Format the template with the input variables.
        return DEFAULT_FORMATTER_MAPPING[self.template_format](
            self.template, **full_kwargs
        )

    # generalization to accommodate abitrary-type input variables
    def _merge_partial_and_user_variables(self, **kwargs: Any) -> Dict[str, Any]:
        partial_kwargs = {
            k: v() if callable(v) else v for k, v in self.partial_variables.items()
        }
        return {**partial_kwargs, **kwargs}

    @property
    def _prompt_type(self) -> str:
        return "convertor-prompt-template"
