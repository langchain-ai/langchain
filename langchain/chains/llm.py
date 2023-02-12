"""Chain that just formats a prompt and calls an LLM."""
from string import Formatter
from typing import Any, Dict, List, Sequence, Union

from pydantic import BaseModel, Extra, validator

from langchain.chains.base import Chain
from langchain.input import get_colored_text
from langchain.llms.base import BaseLLM
from langchain.prompts.base import BasePromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import LLMResult


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
    llm: BaseLLM
    """LLM wrapper to use."""
    output_parsing_mode: str = "validate"
    """Output parsing mode, should be one of `validate`, `off`, `parse`."""
    output_key: str = "text"  #: :meta private:

    @validator("output_parsing_mode")
    def valid_output_parsing_mode(cls, v: str) -> str:
        """Validate output parsing mode."""
        _valid_modes = {"off", "validate", "parse"}
        if v not in _valid_modes:
            raise ValueError(
                f"Got `{v}` for output_parsing_mode, should be one of {_valid_modes}"
            )
        return v

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
        return [self.output_key]

    def generate(self, input_list: List[Dict[str, Any]]) -> LLMResult:
        """Generate LLM result from inputs."""
        stop = None
        if "stop" in input_list[0]:
            stop = input_list[0]["stop"]
        prompts = []
        for inputs in input_list:
            selected_inputs = {k: inputs[k] for k in self.prompt.input_variables}
            prompt = self.prompt.format(**selected_inputs)
            _colored_text = get_colored_text(prompt, "green")
            _text = "Prompt after formatting:\n" + _colored_text
            self.callback_manager.on_text(_text, end="\n", verbose=self.verbose)
            if "stop" in inputs and inputs["stop"] != stop:
                raise ValueError(
                    "If `stop` is present in any inputs, should be present in all."
                )
            prompts.append(prompt)
        response = self.llm.generate(prompts, stop=stop)

        return response

    def _parse_llm_outputs(self, response: LLMResult) -> List[dict]:
        outputs = []
        _should_parse = self.output_parsing_mode != "off"
        for generation in response.generations:
            # Get the text of the top generated string.
            response_item = generation[0].text
            if self.prompt.output_parser is not None and _should_parse:
                try:
                    parsed_output = self.prompt.output_parser.parse(response_item)
                except Exception as e:
                    raise ValueError("Output of LLM not as expected") from e
                if self.output_parsing_mode == "parse":
                    response_item = parsed_output
            outputs.append({self.output_key: response_item})
        return outputs

    def apply(self, input_list: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Utilize the LLM generate method for speed gains."""
        response = self.generate(input_list)
        outputs = self._parse_llm_outputs(response)
        return outputs

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        return self.apply([inputs])[0]

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

    def predict_and_parse(self, **kwargs: Any) -> Union[str, List[str], Dict[str, str]]:
        """Call predict and then parse the results."""
        result = self.predict(**kwargs)
        if self.prompt.output_parser is not None:
            return self.prompt.output_parser.parse(result)
        else:
            return result

    def apply_and_parse(
        self, input_list: List[Dict[str, Any]]
    ) -> Sequence[Union[str, List[str], Dict[str, str]]]:
        """Call apply and then parse the results."""
        result = self.apply(input_list)
        if self.prompt.output_parser is not None:
            new_result = []
            for res in result:
                text = res[self.output_key]
                new_result.append(self.prompt.output_parser.parse(text))
            return new_result
        else:
            return result

    @property
    def _chain_type(self) -> str:
        return "llm_chain"

    @classmethod
    def from_string(cls, llm: BaseLLM, template: str) -> Chain:
        """Create LLMChain from LLM and template."""
        input_variables = {
            v for _, v, _, _ in Formatter().parse(template) if v is not None
        }
        prompt_template = PromptTemplate(
            input_variables=list(input_variables), template=template
        )
        return cls(llm=llm, prompt=prompt_template)
