"""Base schema types."""
from typing import Dict, List

from pydantic import BaseModel, Extra, root_validator

from langchain.formatting import formatter


class Prompt(BaseModel):
    """Schema to represent a prompt for an LLM."""

    input_variables: List[str]
    template: str

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator()
    def template_is_valid(cls, values: Dict) -> Dict:
        """Check that template and input variables are consistent."""
        input_variables = values["input_variables"]
        template = values["template"]
        dummy_inputs = {input_variable: "foo" for input_variable in input_variables}
        try:
            formatter.format(template, **dummy_inputs)
        except KeyError:
            raise ValueError("Invalid prompt schema.")
        return values
