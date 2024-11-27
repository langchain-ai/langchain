from typing import Any, Callable, Union
from typing import Optional as Optional

from pydantic import model_validator

from langchain_core.prompt_values import PromptValue
from langchain_core.prompts.base import BasePromptTemplate
from langchain_core.prompts.chat import BaseChatPromptTemplate


def _get_inputs(inputs: dict, input_variables: list[str]) -> dict:
    ret = {}
    for k in input_variables:
        if k in inputs:
            ret[k] = inputs[k]
    return ret


class PipelinePromptTemplate(BasePromptTemplate):
    """Prompt template for composing multiple prompt templates together.

    This can be useful when you want to reuse parts of prompts.

    A PipelinePrompt consists of two main parts:
        - final_prompt: This is the final prompt that is returned
        - pipeline_prompts: This is a list of tuples, consisting
          of a string (`name`) and a Prompt Template.
          Each PromptTemplate will be formatted and then passed
          to future prompt templates as a variable with
          the same name as `name`
    """

    final_prompt: BasePromptTemplate
    """The final prompt that is returned."""
    pipeline_prompts: list[tuple[str, BasePromptTemplate]]
    """A list of tuples, consisting of a string (`name`) and a Prompt Template."""

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "prompts", "pipeline"]

    @model_validator(mode="before")
    @classmethod
    def get_input_variables(cls, values: dict) -> Any:
        """Get input variables."""
        created_variables = set()
        all_variables = set()
        for k, prompt in values["pipeline_prompts"]:
            created_variables.add(k)
            all_variables.update(prompt.input_variables)
        values["input_variables"] = list(all_variables.difference(created_variables))
        return values

    def format_prompt(self, **kwargs: Any) -> PromptValue:
        """Format the prompt with the inputs.

        Args:
            kwargs: Any arguments to be passed to the prompt template.

        Returns:
            A formatted string.
        """
        for k, prompt in self.pipeline_prompts:
            _inputs = _get_inputs(kwargs, prompt.input_variables)
            if isinstance(prompt, BaseChatPromptTemplate):
                kwargs[k] = prompt.format_messages(**_inputs)
            else:
                kwargs[k] = prompt.format(**_inputs)
        _inputs = _get_inputs(kwargs, self.final_prompt.input_variables)
        return self.final_prompt.format_prompt(**_inputs)

    async def aformat_prompt(self, **kwargs: Any) -> PromptValue:
        """Async format the prompt with the inputs.

        Args:
            kwargs: Any arguments to be passed to the prompt template.

        Returns:
            A formatted string.
        """
        for k, prompt in self.pipeline_prompts:
            _inputs = _get_inputs(kwargs, prompt.input_variables)
            if isinstance(prompt, BaseChatPromptTemplate):
                kwargs[k] = await prompt.aformat_messages(**_inputs)
            else:
                kwargs[k] = await prompt.aformat(**_inputs)
        _inputs = _get_inputs(kwargs, self.final_prompt.input_variables)
        return await self.final_prompt.aformat_prompt(**_inputs)

    def format(self, **kwargs: Any) -> str:
        """Format the prompt with the inputs.

        Args:
            kwargs: Any arguments to be passed to the prompt template.

        Returns:
            A formatted string.
        """
        return self.format_prompt(**kwargs).to_string()

    async def aformat(self, **kwargs: Any) -> str:
        """Async format the prompt with the inputs.

        Args:
            kwargs: Any arguments to be passed to the prompt template.

        Returns:
            A formatted string.
        """
        return (await self.aformat_prompt(**kwargs)).to_string()

    # Ignoring the type below since partial makes modifications rather
    # than returning a new template
    def partial(self, **kwargs: Union[str, Callable[[], str]]) -> None:  # type: ignore[override]
        """Add partial arguments to prompts in pipeline_prompts

        Args:
            kwargs: dict[str, str], partial variables to set.
        """
        for i, string_and_prompt in enumerate(self.pipeline_prompts):
            k, prompt = string_and_prompt
            prompt_kwargs = {}
            for partial_var, partial_input in kwargs.items():
                if partial_var in prompt.input_variables:
                    prompt_kwargs[partial_var] = partial_input
            if len(prompt_kwargs) > 0:
                self.pipeline_prompts[i] = (k, prompt.partial(**prompt_kwargs))

    @property
    def _prompt_type(self) -> str:
        raise ValueError


PipelinePromptTemplate.model_rebuild()
