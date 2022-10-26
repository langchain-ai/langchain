"""Chain that implements the ReAct paper from https://arxiv.org/pdf/2210.03629.pdf."""
import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Extra

from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chains.react.prompt import PROMPT
from langchain.llms.base import LLM


class PageWithLookups(BaseModel):
    """Class to hold a text page that one can lookup terms in."""

    page_content: str
    lookup_str: str = ""
    lookup_index = 0

    @property
    def paragraphs(self) -> List[str]:
        """Paragraphs of the page."""
        return self.page_content.split("\n\n")

    @property
    def summary(self) -> str:
        """Summary of the page (the first paragraph)."""
        return self.paragraphs[0]

    def lookup(self, string: str) -> str:
        """Lookup a term in the page, imitating cmd-F functionality."""
        if string.lower() != self.lookup_str:
            self.lookup_str = string.lower()
            self.lookup_index = 0
        else:
            self.lookup_index += 1
        lookups = [p for p in self.paragraphs if self.lookup_str in p.lower()]
        if len(lookups) == 0:
            return "No Results"
        elif self.lookup_index >= len(lookups):
            return "No More Results"
        else:
            result_prefix = f"(Result {self.lookup_index + 1}/{len(lookups)})"
            return f"{result_prefix} {lookups[self.lookup_index]}"


def search_wiki_page(search: str) -> Tuple[str, Optional[PageWithLookups]]:
    """Try to search for wiki page.

    If page exists, return the page summary, and a PageWithLookups object.
    If page does not exist, return similar entries.
    """
    try:
        import wikipedia
    except ImportError:
        raise ValueError(
            "Could not import wikipedia python package. "
            "Please it install it with `pip install wikipedia`."
        )
    try:
        page_content = wikipedia.page(search).content
        wiki_page = PageWithLookups(page_content=page_content)
        observation = wiki_page.summary
    except wikipedia.PageError:
        wiki_page = None
        observation = (
            f"Could not find [{search}]. " f"Similar: {wikipedia.search(search)}"
        )
    except wikipedia.DisambiguationError:
        wiki_page = None
        observation = (
            f"Could not find [{search}]. " f"Similar: {wikipedia.search(search)}"
        )
    return observation, wiki_page


class ActionError(Exception):
    """An error to raise when there is no action suggested."""


def predict_until_observation(
    llm_chain: LLMChain, prompt: str, i: int
) -> Tuple[str, str, str]:
    """Generate text until an observation is needed."""
    action_prefix = f"Action {i}: "
    stop_seq = f"\nObservation {i}:"
    ret_text = llm_chain.predict(input=prompt, stop=[stop_seq])
    while not ret_text.split("\n")[-1].startswith(action_prefix):
        ret_text += f"\nAction {i}:"
        new_text = llm_chain.predict(input=prompt + ret_text, stop=[stop_seq])
        ret_text += new_text
    action_block = ret_text.split("\n")[-1]
    action_str = action_block[len(action_prefix) :]
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
        prompt = f"{question}\nThought 1:"
        i = 1
        wiki_page = None
        while True:
            ret_text, action, _input = predict_until_observation(llm_chain, prompt, i)
            prompt += ret_text
            print(action, _input)
            if action == "Search":
                observation, wiki_page = search_wiki_page(_input)
                print(observation)
            elif action == "Lookup":
                if wiki_page is None:
                    raise ValueError("Cannot lookup without a successful search first")
                observation = wiki_page.lookup(_input)
            elif action == "Finish":
                return {"full_logic": prompt, self.output_key: _input}
            else:
                raise ValueError(f"Got unknown action directive: {action}")
            prompt += f"\nObservation {i}: " + observation + f"\nThought {i + 1}:"
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
