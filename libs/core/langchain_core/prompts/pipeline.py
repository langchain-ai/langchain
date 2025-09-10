"""[DEPRECATED] Pipeline prompt template."""

from typing import Any

from pydantic import model_validator

from langchain_core._api.deprecation import deprecated
from langchain_core.prompt_values import PromptValue
from langchain_core.prompts.base import BasePromptTemplate
from langchain_core.prompts.chat import BaseChatPromptTemplate


def _get_inputs(inputs: dict, input_variables: list[str]) -> dict:
    return {k: inputs[k] for k in input_variables}


@deprecated(
    since="0.3.22",
    removal="1.0",
    message=(
        "This class is deprecated in favor of chaining individual prompts together."
    ),
)
class PipelinePromptTemplate(BasePromptTemplate):
    """Pipeline prompt template.

    This has been deprecated in favor of chaining individual prompts together in your
    code; e.g. using a for loop, you could do:

    .. code-block:: python

        my_input = {"key": "value"}
        for name, prompt in pipeline_prompts:
            my_input[name] = prompt.invoke(my_input).to_string()
        my_output = final_prompt.invoke(my_input)

    Prompt template for composing multiple prompt templates together.

    This can be useful when you want to reuse parts of prompts.

    A PipelinePrompt consists of two main parts:

    - final_prompt: This is the final prompt that is returned
    - pipeline_prompts: This is a list of tuples, consisting
      of a string (``name``) and a Prompt Template.
      Each PromptTemplate will be formatted and then passed
      to future prompt templates as a variable with
      the same name as ``name``

    """

    final_prompt: BasePromptTemplate
    """The final prompt that is returned."""
    pipeline_prompts: list[tuple[str, BasePromptTemplate]]
    """A list of tuples, consisting of a string (``name``) and a Prompt Template."""

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the langchain object.

        Returns:
            ``["langchain", "prompts", "pipeline"]``
        """
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
            inputs = _get_inputs(kwargs, prompt.input_variables)
            if isinstance(prompt, BaseChatPromptTemplate):
                kwargs[k] = prompt.format_messages(**inputs)
            else:
                kwargs[k] = prompt.format(**inputs)
        inputs = _get_inputs(kwargs, self.final_prompt.input_variables)
        return self.final_prompt.format_prompt(**inputs)

    async def aformat_prompt(self, **kwargs: Any) -> PromptValue:
        """Async format the prompt with the inputs.

        Args:
            kwargs: Any arguments to be passed to the prompt template.

        Returns:
            A formatted string.
        """
        for k, prompt in self.pipeline_prompts:
            inputs = _get_inputs(kwargs, prompt.input_variables)
            if isinstance(prompt, BaseChatPromptTemplate):
                kwargs[k] = await prompt.aformat_messages(**inputs)
            else:
                kwargs[k] = await prompt.aformat(**inputs)
        inputs = _get_inputs(kwargs, self.final_prompt.input_variables)
        return await self.final_prompt.aformat_prompt(**inputs)

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

    @property
    def _prompt_type(self) -> str:
        raise ValueError
