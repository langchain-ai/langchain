"""Chain that implements the ReAct paper from https://arxiv.org/pdf/2210.03629.pdf."""
import re
from typing import Any, Dict, List, Tuple

from pydantic import BaseModel, Extra

from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chains.react.prompt import PROMPT
from langchain.docstore.base import Docstore
from langchain.docstore.document import Document
from langchain.input import ChainedInput
from langchain.llms.base import LLM


def predict_until_observation(
    llm_chain: LLMChain, prompt: str, i: int
) -> Tuple[str, str, str]:
    """Generate text until an observation is needed."""
    action_prefix = f"Action {i}: "
    stop_seq = f"\nObservation {i}:"
    ret_text = llm_chain.predict(input=prompt, stop=[stop_seq])
    # Sometimes the LLM forgets to take an action, so we prompt it to.
    while not ret_text.split("\n")[-1].startswith(action_prefix):
        ret_text += f"\nAction {i}:"
        new_text = llm_chain.predict(input=prompt + ret_text, stop=[stop_seq])
        ret_text += new_text
    # The action block should be the last line.
    action_block = ret_text.split("\n")[-1]
    action_str = action_block[len(action_prefix) :]
    # Parse out the action and the directive.
    re_matches = re.search(r"(.*?)\[(.*?)\]", action_str)
    if re_matches is None:
        raise ValueError(f"Could not parse action directive: {action_str}")
    return ret_text, re_matches.group(1), re_matches.group(2)


class ReActChain(Chain, BaseModel):
    """Chain that implements the ReAct paper.

    Example:
        .. code-block:: python

            from langchain import ReActChain, OpenAI
            react = ReAct(llm=OpenAI())
    """

    llm: LLM
    """LLM wrapper to use."""
    docstore: Docstore
    """Docstore to use."""
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
        return ["full_logic", self.output_key]

    def _run(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        question = inputs[self.input_key]
        llm_chain = LLMChain(llm=self.llm, prompt=PROMPT)
        chained_input = ChainedInput(f"{question}\nThought 1:", verbose=self.verbose)
        i = 1
        document = None
        while True:
            ret_text, action, directive = predict_until_observation(
                llm_chain, chained_input.input, i
            )
            chained_input.add(ret_text, color="green")
            if action == "Search":
                result = self.docstore.search(directive)
                if isinstance(result, Document):
                    document = result
                    observation = document.summary
                else:
                    document = None
                    observation = result
            elif action == "Lookup":
                if document is None:
                    raise ValueError("Cannot lookup without a successful search first")
                observation = document.lookup(directive)
            elif action == "Finish":
                return {"full_logic": chained_input.input, self.output_key: directive}
            else:
                raise ValueError(f"Got unknown action directive: {action}")
            chained_input.add(f"\nObservation {i}: ")
            chained_input.add(observation, color="yellow")
            chained_input.add(f"\nThought {i + 1}:")
            i += 1

    def run(self, question: str) -> str:
        """Run ReAct framework.

        Args:
            question: Question to be answered.

        Returns:
            Final answer from thinking through the ReAct framework.

        Example:
            .. code-block:: python

                question = "Were Scott Derrickson and Ed Wood of the same nationality?"
                answer = react.run(question)
        """
        return self({self.input_key: question})[self.output_key]
