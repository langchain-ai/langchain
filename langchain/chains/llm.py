"""Chain that just formats a prompt and calls an LLM."""
from types import FunctionType
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from pydantic import BaseModel, Extra

from langchain.chains.base import Chain
from langchain.chains.llm_validator import LLMCorrectChain
from langchain.input import get_colored_text
from langchain.llms.base import BaseLLM
from langchain.prompts.base import BaseOutputParser, BasePromptTemplate
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
    parser_retries: int = 0
    """Number of times to retry the query if it fails."""
    parser_correct_on_error: bool = False
    """If True, will try to correct the input if the query fails."""
    output_key: str = "text"  #: :meta private:

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
        prompts, stop = self.prep_prompts(input_list)
        response = self.llm.generate(prompts, stop=stop)
        return response

    async def agenerate(self, input_list: List[Dict[str, Any]]) -> LLMResult:
        """Generate LLM result from inputs."""
        prompts, stop = self.prep_prompts(input_list)
        response = await self.llm.agenerate(prompts, stop=stop)
        return response

    def prep_prompts(
        self, input_list: List[Dict[str, Any]]
    ) -> Tuple[List[str], Optional[List[str]]]:
        """Prepare prompts from inputs."""
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
        return prompts, stop

    def apply(self, input_list: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Utilize the LLM generate method for speed gains."""
        response = self.generate(input_list)
        return self.create_outputs(response)

    async def aapply(self, input_list: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Utilize the LLM generate method for speed gains."""
        response = await self.agenerate(input_list)
        return self.create_outputs(response)

    def create_outputs(self, response: LLMResult) -> List[Dict[str, str]]:
        """Create outputs from response."""
        outputs = []
        for generation in response.generations:
            # Get the text of the top generated string.
            response_str = generation[0].text
            outputs.append({self.output_key: response_str})
        return outputs

    def _call(self, inputs: Dict[str, Any], parse: bool = True) -> Dict[str, str]:
        if not parse or self.prompt.output_parser is None:
            return self.apply([inputs])[0]

        parse_last_exception: Optional[Exception] = None
        for _ in range(1 + self.parser_retries):
            result = self.apply([inputs])[0]
            text = result[self.output_key]
            try:
                result[self.output_key] = self.prompt.parse_output(text)
                return result
            except Exception as exc:
                parse_last_exception = exc

                if self.parser_correct_on_error:
                    corrected = self._parser_correct_on_error(text, exc)
                    try:
                        result[self.output_key] = self.prompt.parse_output(corrected)
                        return result
                    except Exception:
                        pass

        if parse_last_exception is not None:
            raise parse_last_exception
        raise ValueError("parser_retries shouldn't be negative.")

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
        return self(kwargs, parse=False)[self.output_key]

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

    def predict_and_parse(self, **kwargs: Any) -> Union[str, List[str], Dict[str, str]]:
        """Call predict and then parse the results."""
        return self(kwargs)[self.output_key]

    def apply_and_parse(
        self, input_list: List[Dict[str, Any]]
    ) -> Sequence[Union[str, List[str], Dict[str, str]]]:
        """Call apply and then parse the results."""
        result = self.apply(input_list)
        if self.prompt.output_parser is not None:
            new_result = []
            for res in result:
                text = res[self.output_key]
                new_result.append(self.prompt.parse_output(text))
            return new_result
        return result

    def _parser_correct_on_error(self, text: str, e: Exception) -> str:
        """Try to correct the input if the parser function fails."""
        llm_correct_validator = LLMCorrectChain(llm=self.llm)

        if isinstance(self.prompt.output_parser, BaseOutputParser):
            parser_name = self.prompt.output_parser.__class__.__name__
        elif isinstance(self.prompt.output_parser, FunctionType):
            parser_name = self.prompt.output_parser.__name__
        else:
            raise ValueError("Could not determine output parser name.")

        response = llm_correct_validator(
            {
                "text": text,
                "validator_name": parser_name,
                "error_message": str(e),
            }
        )
        corrected = response["corrected"]
        return corrected

    @property
    def _chain_type(self) -> str:
        return "llm_chain"

    @classmethod
    def from_string(cls, llm: BaseLLM, template: str) -> Chain:
        """Create LLMChain from LLM and template."""
        prompt_template = PromptTemplate.from_template(template)
        return cls(llm=llm, prompt=prompt_template)
