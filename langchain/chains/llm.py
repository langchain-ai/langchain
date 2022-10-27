"""Chain that just formats a prompt and calls an LLM."""
from collections import defaultdict
import math
from typing import Any, Callable, Dict, List, Tuple

from pydantic import BaseModel, Extra

from langchain.chains.base import Chain
from langchain.llms.base import LLM
from langchain.prompt import Prompt
import re


class LLMChain(Chain, BaseModel):
    """Chain to run queries against LLMs.

    Example:
        .. code-block:: python

            from langchain import LLMChain, OpenAI, Prompt
            prompt_template = "Tell me a {adjective} joke"
            prompt = Prompt(input_variables=["adjective"], template=prompt_template)
            llm = LLMChain(llm=OpenAI(), prompt=prompt)
    """

    prompt: Prompt
    """Prompt object to use."""
    llm: LLM
    """LLM wrapper to use."""
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

    def _run(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        selected_inputs = {k: inputs[k] for k in self.prompt.input_variables}
        prompt = self.prompt.format(**selected_inputs)

        kwargs = {}
        if "stop" in inputs:
            kwargs["stop"] = inputs["stop"]
        response = self.llm(prompt, **kwargs)
        return {self.output_key: response}

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


class ChainOfThoughtParser(BaseModel):
    """Parser to separate the reasoning steps from the answer."""

    reasoning_parser: Callable[[str], str]
    """Function to parse the reasoning steps from the generated text."""
    answer_parser: Callable[[str], str]
    """Function to parse the answer from the generated text."""

    def parse_completion(self, text: str) -> Tuple[str, str]:
        """Parse the reasoning steps and answer from the completion."""
        reasoning = self.reasoning_parser(text)
        answer = self.answer_parser(text)
        return reasoning, answer


# Default parser returns the string preceding "The answer is" (case invariant) as the reasoning
# and the string following as the answer.

_UNKNOWN_ANSWER = "I don't know."

def _default_answer_parser(text: str) -> str:
    """Default answer parser."""
    try:
        # Use re to split the text along "The answer is" (case invariant) and return the second
        # element of the resulting list.
        return re.split(r"(?i)the\sanswer\sis", text)[1].strip()
    except IndexError:
        return _UNKNOWN_ANSWER


def _default_reasoning_parser(text: str) -> str:
    """Default reasoning parser."""
    try:
        return re.split(r"(?i)the\sanswer\sis", text)[0].strip()
    except IndexError:
        return text


DEFAULT_CHAIN_OF_THOUGHT_PARSER = ChainOfThoughtParser(
    reasoning_parser=_default_reasoning_parser, answer_parser=_default_answer_parser
)


class SelfConsistencyLLMChain(LLMChain, BaseModel):
    """LLM Chain that uses self-consistency to improve the reliability of its outputs."""

    parser: ChainOfThoughtParser = DEFAULT_CHAIN_OF_THOUGHT_PARSER
    """Parser to separate the reasoning steps from the answer."""
    max_iterations: int = 5
    """Maximum number of iterations to run."""
    normalize_probs: bool = True

    def _run(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """Run the chain."""
        selected_inputs = {k: inputs[k] for k in self.prompt.input_variables}
        prompt = self.prompt.format(**selected_inputs)

        kwargs = {}
        if "stop" in inputs:
            kwargs["stop"] = inputs["stop"]
        answers = defaultdict(float)
        responses = defaultdict(list)
        n = 0
        while n < self.max_iterations:
            _responses = self.llm.generate(prompt, **kwargs)
            for response in _responses:
                reasoning, answer = self.parser.parse_completion(response.text)
                if response.logprobs is not None:
                    total_logprob = sum(response.logprobs)
                    if self.normalize_probs:
                        total_logprob /= len(response.logprobs)
                    generated_prob = math.exp(total_logprob)
                else:
                    generated_prob = 1.0
                answers[answer] += generated_prob
                responses[answer].append((reasoning, answer, generated_prob))
                n += 1
        answer = max(answers, key=answers.get)
        sorted_answers = sorted(responses[answer], key=lambda x: x[2], reverse=True)
        if answer == _UNKNOWN_ANSWER:
            # If the model doesn't know, output the related reasoning steps.
            flipped_response = sorted_answers[0][0]
        else:
            flipped_response = answer
        return {self.output_key: flipped_response}
