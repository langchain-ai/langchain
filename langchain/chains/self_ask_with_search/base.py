"""Chain that does self ask with search."""
from typing import Any, Dict, List

from pydantic import BaseModel, Extra

from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chains.self_ask_with_search.prompt import PROMPT
from langchain.chains.serpapi import SerpAPIChain
from langchain.input import ChainedInput
from langchain.llms.base import LLM


def extract_answer(generated: str) -> str:
    """Extract answer from text."""
    if "\n" not in generated:
        last_line = generated
    else:
        last_line = generated.split("\n")[-1]

    if ":" not in last_line:
        after_colon = last_line
    else:
        after_colon = generated.split(":")[-1]

    if " " == after_colon[0]:
        after_colon = after_colon[1:]
    if "." == after_colon[-1]:
        after_colon = after_colon[:-1]

    return after_colon


def extract_question(generated: str, followup: str) -> str:
    """Extract question from text."""
    if "\n" not in generated:
        last_line = generated
    else:
        last_line = generated.split("\n")[-1]

    if followup not in last_line:
        print("we probably should never get here..." + generated)

    if ":" not in last_line:
        after_colon = last_line
    else:
        after_colon = generated.split(":")[-1]

    if " " == after_colon[0]:
        after_colon = after_colon[1:]
    if "?" != after_colon[-1]:
        print("we probably should never get here..." + generated)

    return after_colon


def get_last_line(generated: str) -> str:
    """Get the last line in text."""
    if "\n" not in generated:
        last_line = generated
    else:
        last_line = generated.split("\n")[-1]

    return last_line


def greenify(_input: str) -> str:
    """Add green highlighting to text."""
    return "\x1b[102m" + _input + "\x1b[0m"


def yellowfy(_input: str) -> str:
    """Add yellow highlighting to text."""
    return "\x1b[106m" + _input + "\x1b[0m"


class SelfAskWithSearchChain(Chain, BaseModel):
    """Chain that does self ask with search.

    Example:
        .. code-block:: python

            from langchain import SelfAskWithSearchChain, OpenAI, SerpAPIChain
            search_chain = SerpAPIChain()
            self_ask = SelfAskWithSearchChain(llm=OpenAI(), search_chain=search_chain)
    """

    llm: LLM
    """LLM wrapper to use."""
    search_chain: SerpAPIChain
    """Search chain to use."""
    input_key: str = "question"  #: :meta private:
    output_key: str = "answer"  #: :meta private:

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Expect input key.

        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Expect output key.

        :meta private:
        """
        return [self.output_key]

    def _run(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        chained_input = ChainedInput(inputs[self.input_key], verbose=self.verbose)
        chained_input.add("\nAre follow up questions needed here:")
        llm_chain = LLMChain(llm=self.llm, prompt=PROMPT)
        intermediate = "\nIntermediate answer:"
        followup = "Follow up:"
        finalans = "\nSo the final answer is:"
        ret_text = llm_chain.predict(input=chained_input.input, stop=[intermediate])
        chained_input.add(ret_text, color="green")
        while followup in get_last_line(ret_text):
            question = extract_question(ret_text, followup)
            external_answer = self.search_chain.search(question)
            if external_answer is not None:
                chained_input.add(intermediate + " ")
                chained_input.add(external_answer + ".", color="yellow")
                ret_text = llm_chain.predict(
                    input=chained_input.input, stop=["\nIntermediate answer:"]
                )
                chained_input.add(ret_text, color="green")
            else:
                # We only get here in the very rare case that Google returns no answer.
                chained_input.add(intermediate + " ")
                preds = llm_chain.predict(
                    input=chained_input.input, stop=["\n" + followup, finalans]
                )
                chained_input.add(preds, color="green")

        if finalans not in ret_text:
            chained_input.add(finalans)
            ret_text = llm_chain.predict(input=chained_input.input, stop=["\n"])
            chained_input.add(ret_text, color="green")

        return {self.output_key: ret_text}

    def run(self, question: str) -> str:
        """Run self ask with search chain.

        Args:
            question: Question to run self-ask-with-search with.

        Returns:
            The final answer

        Example:
            .. code-block:: python

                answer = selfask.run("What is the capital of Idaho?")
        """
        return self({self.input_key: question})[self.output_key]
