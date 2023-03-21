"""Chain that just formats a prompt and calls an LLM."""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from pydantic import BaseModel, Extra, root_validator

from langchain.chains.base import Chain
from langchain.input import get_colored_text
from langchain.prompts.base import BasePromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import (
    BaseLanguageModel,
    BaseOutputParser,
    Generation,
    LLMResult,
    PromptValue,
)


class LLMChain(Chain, BaseModel):
    """Chain to run queries against LLMs.

    Example:
        .. code-block:: python

            from langchain import LLMChain, OpenAI, PromptTemplate
            prompt_template = "Tell me a {adjective} joke"
            prompt = PromptTemplate(
                input_variables=["adjective"], template=prompt_template
            )
            llm = LLMChain(llm=OpenAI(), prompt=prompt)
    """

    prompt: BasePromptTemplate
    """Prompt object to use."""
    llm: BaseLanguageModel
    output_key: str = "text"  #: :meta private:
    output_parser: Optional[BaseOutputParser] = None
    debug: bool = False

    @root_validator()
    def validate_output_parser(cls, values: Dict) -> Dict:
        """Validate output parser."""
        prompt: BasePromptTemplate = values["prompt"]
        output_parser = values["output_parser"]
        if prompt.output_parser is not None:
            warnings.warn(
                "Got an output parser on the prompt "
                "- please transition this to being passed into the LLMChain."
            )
            if output_parser is not None:
                raise ValueError(
                    "Got an output parser on the prompt as well on the LLMChain - "
                    "should only be provided in one place."
                )
            values["output_parser"] = prompt.output_parser
        return values

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Will be whatever keys the prompt expects.

        :meta private:
        """
        return self.prompt.input_variables

    @property
    def output_keys(self) -> List[str]:
        """Will always return text key.

        :meta private:
        """
        if not self.debug:
            return [self.output_key]
        else:
            return [self.output_key, "raw", "error"]

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        return self.apply([inputs])[0]

    def generate(
        self, input_list: List[Dict[str, Any]]
    ) -> Tuple[LLMResult, List[PromptValue]]:
        """Generate LLM result from inputs."""
        prompts, stop = self.prep_prompts(input_list)
        return self.llm.generate_prompt(prompts, stop), prompts

    async def agenerate(
        self, input_list: List[Dict[str, Any]]
    ) -> Tuple[LLMResult, List[PromptValue]]:
        """Generate LLM result from inputs."""
        prompts, stop = await self.aprep_prompts(input_list)
        return await self.llm.agenerate_prompt(prompts, stop), prompts

    def prep_prompts(
        self, input_list: List[Dict[str, Any]]
    ) -> Tuple[List[PromptValue], Optional[List[str]]]:
        """Prepare prompts from inputs."""
        stop = None
        if "stop" in input_list[0]:
            stop = input_list[0]["stop"]
        prompts = []
        for inputs in input_list:
            selected_inputs = {k: inputs[k] for k in self.prompt.input_variables}
            prompt = self.prompt.format_prompt(**selected_inputs)
            _colored_text = get_colored_text(prompt.to_string(), "green")
            _text = "Prompt after formatting:\n" + _colored_text
            self.callback_manager.on_text(_text, end="\n", verbose=self.verbose)
            if "stop" in inputs and inputs["stop"] != stop:
                raise ValueError(
                    "If `stop` is present in any inputs, should be present in all."
                )
            prompts.append(prompt)
        return prompts, stop

    async def aprep_prompts(
        self, input_list: List[Dict[str, Any]]
    ) -> Tuple[List[PromptValue], Optional[List[str]]]:
        """Prepare prompts from inputs."""
        stop = None
        if "stop" in input_list[0]:
            stop = input_list[0]["stop"]
        prompts = []
        for inputs in input_list:
            selected_inputs = {k: inputs[k] for k in self.prompt.input_variables}
            prompt = self.prompt.format_prompt(**selected_inputs)
            _colored_text = get_colored_text(prompt.to_string(), "green")
            _text = "Prompt after formatting:\n" + _colored_text
            if self.callback_manager.is_async:
                await self.callback_manager.on_text(
                    _text, end="\n", verbose=self.verbose
                )
            else:
                self.callback_manager.on_text(_text, end="\n", verbose=self.verbose)
            if "stop" in inputs and inputs["stop"] != stop:
                raise ValueError(
                    "If `stop` is present in any inputs, should be present in all."
                )
            prompts.append(prompt)
        return prompts, stop

    def apply(self, input_list: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Utilize the LLM generate method for speed gains."""
        response, prompts = self.generate(input_list)
        return self.create_outputs(response, prompts)

    async def aapply(self, input_list: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Utilize the LLM generate method for speed gains."""
        response, prompts = await self.agenerate(input_list)
        return self.create_outputs(response, prompts)

    def _get_final_output(
        self, generations: List[Generation], prompt_value: PromptValue
    ) -> Dict:
        """Get the final output from a list of generations for a prompt."""
        completion = generations[0].text
        if self.output_parser is not None:
            try:
                new_completion = self.output_parser.parse_with_prompt(
                    completion, prompt_value
                )
                result = {self.output_key: new_completion}
                if self.debug:
                    result["raw"] = completion
                    result["errors"] = []
            except Exception as e:
                if self.debug:
                    result = {
                        self.output_key: None,
                        "raw": completion,
                        "error": [repr(e)],
                    }
        else:
            result = {self.output_key: completion}
        return result

    def create_outputs(
        self, response: LLMResult, prompts: List[PromptValue]
    ) -> List[Dict[str, str]]:
        """Create outputs from response."""
        return [
            # Get the text of the top generated string.
            self._get_final_output(generation, prompts[i])
            for i, generation in enumerate(response.generations)
        ]

    async def _acall(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        return (await self.aapply([inputs]))[0]

    def predict(self, **kwargs: Any) -> str:
        """Format prompt with kwargs and pass to LLM.

        Args:
            **kwargs: Keys to pass to prompt template.

        Returns:
            Completion from LLM.

        Example:
            .. code-block:: python

                completion = llm.predict(adjective="funny")
        """
        return self(kwargs)[self.output_key]

    async def apredict(self, **kwargs: Any) -> str:
        """Format prompt with kwargs and pass to LLM.

        Args:
            **kwargs: Keys to pass to prompt template.

        Returns:
            Completion from LLM.

        Example:
            .. code-block:: python

                completion = llm.predict(adjective="funny")
        """
        return (await self.acall(kwargs))[self.output_key]

    # If an output_parser is provided, it should always be applied.
    # TODO: remove these methods.
    def predict_and_parse(self, **kwargs: Any) -> Union[str, List[str], Dict[str, str]]:
        """Call predict and then parse the results."""
        return self.predict(**kwargs)

    def apply_and_parse(
        self, input_list: List[Dict[str, Any]]
    ) -> Sequence[Union[str, List[str], Dict[str, str]]]:
        """Call apply and then parse the results."""
        return self.apply(input_list)

    async def aapply_and_parse(
        self, input_list: List[Dict[str, Any]]
    ) -> Sequence[Union[str, List[str], Dict[str, str]]]:
        """Call apply and then parse the results."""
        return await self.aapply(input_list)

    @property
    def _chain_type(self) -> str:
        return "llm_chain"

    @classmethod
    def from_string(cls, llm: BaseLanguageModel, template: str) -> Chain:
        """Create LLMChain from LLM and template."""
        prompt_template = PromptTemplate.from_template(template)
        return cls(llm=llm, prompt=prompt_template)
