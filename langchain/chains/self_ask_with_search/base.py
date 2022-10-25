"""Chain that does self ask with search."""
from typing import Any, Dict, List

from pydantic import BaseModel, Extra

from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chains.self_ask_with_search.prompt import PROMPT
from langchain.chains.serpapi import SerpAPIChain
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
        question = inputs[self.input_key]
        llm_chain = LLMChain(llm=self.llm, prompt=PROMPT)
        intermediate = "\nIntermediate answer:"
        followup = "Follow up:"
        finalans = "\nSo the final answer is:"
        cur_prompt = f"{question}\nAre follow up questions needed here:"
        print(cur_prompt, end="")
        ret_text = llm_chain.predict(input=cur_prompt, stop=[intermediate])
        print(greenify(ret_text), end="")
        while followup in get_last_line(ret_text):
            cur_prompt += ret_text
            question = extract_question(ret_text, followup)
            external_answer = self.search_chain.search(question)
            if external_answer is not None:
                cur_prompt += intermediate + " " + external_answer + "."
                print(
                    intermediate + " " + yellowfy(external_answer) + ".",
                    end="",
                )
                ret_text = llm_chain.predict(
                    input=cur_prompt, stop=["\nIntermediate answer:"]
                )
                print(greenify(ret_text), end="")
            else:
                # We only get here in the very rare case that Google returns no answer.
                cur_prompt += intermediate
                print(intermediate + " ")
                cur_prompt += llm_chain.predict(
                    input=cur_prompt, stop=["\n" + followup, finalans]
                )

        if finalans not in ret_text:
            cur_prompt += finalans
            print(finalans, end="")
            ret_text = llm_chain.predict(input=cur_prompt, stop=["\n"])
            print(greenify(ret_text), end="")

        return {self.output_key: cur_prompt + ret_text}

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
