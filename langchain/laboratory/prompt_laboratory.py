"""Experiment with different prompts on the same model."""
from __future__ import annotations

from typing import Dict, List, Any, Optional, Sequence

from langchain.input import get_color_mapping, print_text
from langchain.llms.base import BaseLLM
from langchain.prompts.prompt import PromptTemplate
from langchain import OpenAI


class PromptLaboratory:
    """Experiment with different prompts on the same model."""

    def __init__(self, llm: BaseLLM, prompts: Sequence[PromptTemplate], names: Optional[List[str]] = None, **kwargs):
        """Initialize with prompts to experiment with.

        Args:
            llm: llm or chat model you want to do experiment on.
            prompts: list of prompts to experiment with.
            kwargs: parameters to initialize llm.
        """
        for prompt in prompts:
            if not isinstance(prompt, PromptTemplate):
                raise ValueError(
                    "PromptLaboratory should now be initialized with PromptTemplate. "
                    "If you want to initialize with Strings, use the `from_templates` method "
                    "instead (`PromptLaboratory.from_templates(...)`)"
                )
        base_input_variables = prompts[0].input_variables
        if False in [set(base_input_variables) == set(prompt.input_variables) for prompt in prompts]:
            raise ValueError("input_variables must be same for every prompt.")
        
        if names is not None:
            if len(names) != len(prompts):
                raise ValueError("Length of prompts does not match length of names.")
        self.llm = llm
        self.prompts = prompts
        self.input_variables = set(base_input_variables)
        prompt_range = [str(i) for i in range(len(self.prompts))]
        self.prompt_colors = get_color_mapping(prompt_range)
        self.names = names

    @classmethod
    def from_templates(
        cls, llm: BaseLLM, templates: Sequence[str], names: Optional[List[str]] = None, **kwargs
    ) -> PromptLaboratory:
        """Initialize with Templates to experiment with and optional llm (defalut: OpenAI()).

        Args:
            templates: list of Strings to experiment with
            llm: llm or chat model you want to do experiment on.
        """
        if llm is None:
            llm = OpenAI(**kwargs)
        prompts = [PromptTemplate.from_template(template) for template in templates]
        if names is not None:
            if len(names) != len(prompts):
                raise ValueError("Length of prompts does not match length of names.")
        else:
            names = ["PROMPT_" + str(i) for i in range(0, len(templates))]
        return cls(prompts, names, llm)

    def compare(self, input_variables: Dict[str, Any]) -> None:
        """Compare model outputs on input variables.

        Args:
            input_variables: dictionary of input variables to experiment all prompts.
        """
        if set(input_variables.keys()) != self.input_variables:
            raise ValueError(
                    f"input_variables mismatch. Expected {self.input_variables}, "
                    f"got {set(input_variables.keys())}"
                )
        print(f"\033[1mInput:\033[0m\n{input_variables}\n")
        for i, prompt in enumerate(self.prompts):
            if self.names is not None:
                name = self.names[i]
            else:
                name = "PROMPT_" + str(i)
            print_text(name, end="\n")
            output = self.llm.generate_prompt([prompt.format_prompt(**input_variables)]).generations[0]
            output = "\n".join([generation.text for generation in output])
            print_text(output, color=self.prompt_colors[str(i)], end="\n\n")