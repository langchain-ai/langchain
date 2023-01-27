"""Prompt building for GetMultipleOutputsChain."""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional

from pydantic import BaseModel, Extra

from langchain.chains.multiple_outputs.config import VariableConfig
from langchain.prompts.base import DictOutputParser
from langchain.prompts.prompt import PromptTemplate


class MultipleOutputsPrompter(BaseModel):
    """Handles the prompting logic for GetMultipleOutputsChain."""

    prefix: str
    """The preamble before we ask the LLM to fill in each variable value."""
    variables: List[VariableConfig]
    """Settings for how each variable is to be displayed and processed."""
    output_parser: Optional[DictOutputParser] = None
    """How to parse the output of calling an LLM on this formatted prompt.

    Must be provided if the LLM is completing all the inputs all at once."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    def __init__(
        self,
        prefix: str,
        variables: Optional[Dict[str, str]] = None,
        variable_configs: Optional[List[VariableConfig]] = None,
        auto_suffix_variable_display: bool = True,
        output_parser: Optional[DictOutputParser] = None,
    ):
        r"""Prompt template creator for getting multiple outputs from the LLM.

        Args:
            prefix: Prompt that asks for the values of each of the variables the LLM
                    should fill in
            variables: A mapping from the output key of each variable to its display in
                    the prompt. Use this instead of `variable_configs` if you don't care
                    about customizing per-variable prompts.
            variable_configs: Information about each variable we want the value of. Use
                    this instead of `variables` if you want fine-grained control over
                    the display of variables in the prompt.
            auto_suffix_variable_display: If true, for every variable that doesn't
                    already have a suffix, the display values provided for each
                    variable will have a colon and the variable stop appended to the
                    end.

                    For example, if this is true, and your variable display is
                    "Action Input", this will automatically suffix the stop to get
                    "Action Input: \"" for the final prompt.
            output_parser: The output parser that will be called if this is asked to
                    let the LLM complete all the outputs at once.
        """
        if variables is None and variable_configs is None:
            raise ValueError("Please set either variables or variable_configs")
        elif variables and variable_configs:
            raise ValueError("Please set only one of variables or variable_configs")

        if not variable_configs:
            # this assert helps tell pydantic `variables` is assured to have a value at
            # this point
            assert variables is not None, "Please put the ValueError's back in"
            variable_configs = [
                VariableConfig(display=display, output_key=output_key)
                for output_key, display in variables.items()
            ]

        if auto_suffix_variable_display:
            new_configs = []
            for variable in variable_configs:
                if variable.display_suffix is None:
                    new_suffix = f": {variable.stop}"
                else:
                    new_suffix = variable.display_suffix  # keep the old one
                new_configs.append(
                    dataclasses.replace(variable, display_suffix=new_suffix)
                )
            variable_configs = new_configs

        super().__init__(
            prefix=prefix, variables=variable_configs, output_parser=output_parser
        )

    @property
    def output_keys(self) -> List[str]:
        """Output keys from this whole chain."""
        return [v.output_key for v in self.variables]

    def prompt_template_for_variable_at(self, i: int) -> PromptTemplate:
        """Prompt for a single output variable to be filled in."""
        output = self.variables[i]
        inputs = self.variables[:i]
        input_keys = [v.output_key for v in inputs]

        template = self.prefix
        for input in inputs:
            template += input.prompt_with_value + "\n"
        template += output.prompt

        return PromptTemplate(
            input_variables=input_keys,
            template=template,
        )

    def prompt_template_for_full_input(self) -> PromptTemplate:
        """Prompt for all outputs to be filled in in a single step."""
        if not self.output_parser:
            raise ValueError(
                "Can't prompt for full input if we don't know how to parse it!"
            )

        first_template = self.prompt_template_for_variable_at(0)
        first_template.output_parser = self.output_parser
        return first_template

    def log(self, completions: Dict[str, str]) -> str:
        """Output the entirety of the LLM's inputs in this chain."""
        initial_prompt = self.prompt_template_for_variable_at(0).format()

        final_prompt = self.prefix
        for var in self.variables:
            final_prompt += var.prompt_with_value + "\n"
        final_prompt = final_prompt.format(**completions)

        assert final_prompt.startswith(initial_prompt), "Prompt reconstruction failed"

        return final_prompt[len(initial_prompt) :].strip()
