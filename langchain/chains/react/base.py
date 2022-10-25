"""Chain that implements the ReAct Methodology."""
from typing import Any, Dict, List

import wikipedia
from pydantic import BaseModel, Extra

from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chains.react.prompt import PROMPT
from langchain.llms.base import LLM


class PageWithLookups(BaseModel):

    page_content: str
    lookup_str: str = ""
    lookup_index = 0

    @property
    def paragraphs(self):
        return self.page_content.split("\n\n")

    @property
    def summary(self):
        return self.paragraphs[0]

    def lookup(self, string):
        if string != self.lookup_str:
            self.lookup_str = string.lower()
            self.lookup_index = 0
        else:
            self.lookup_index += 1
        lookups = [p for p in self.paragraphs if self.lookup_str in p.lower()]
        if len(lookups) == 0:
            return "No Results"
        elif self.lookup_index >= len(lookups):
            return "No Results"
        else:
            result_prefix = f"(Result {self.lookup_index + 1}/{len(lookups)})"
            return f"{result_prefix} {lookups[self.lookup_index]}"



import re


class ActionError(Exception):
    """An error to raise when there is no action suggested."""


def extract_action(text, i):
    action_block = text.split("\n")[-1]
    action_prefix = f"Action {i}: "
    if not action_block.startswith(action_prefix):
        raise ActionError
    action_str = action_block[len(action_prefix) :]
    re_matches = re.search("(.*?)\[(.*?)\]", action_str)
    return re_matches.group(1), re_matches.group(2)


class ReActChain(Chain, BaseModel):
    llm: LLM
    input_key: str = "question"
    output_key: str = "answer"

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Expect input key."""
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Expect output key."""
        return ["full_logic", self.output_key]

    def _run(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        question = inputs[self.input_key]
        llm_chain = LLMChain(llm=self.llm, prompt=PROMPT)
        prompt = f"{question}\nThought 1:"
        i = 1
        stop_seq = f"\nObservation {i}:"
        prefix = ""
        wiki_page = None
        while True:
            ret_text = llm_chain.predict(input=prompt, stop=[stop_seq])
            prompt += ret_text
            try:
                action, _input = extract_action(prefix + ret_text, i)
                print(action, _input)
                if action == "Search":
                    try:
                        page_content = wikipedia.page(_input).content
                        wiki_page = PageWithLookups(page_content=page_content)
                        observation = wiki_page.summary
                    except wikipedia.PageError:
                        wiki_page = None
                        observation = f"Could not find [{_input}]. Similar: {wikipedia.search(_input)}"
                    except wikipedia.DisambiguationError:
                        wiki_page = None
                        observation = f"Could not find [{_input}]. Similar: {wikipedia.search(_input)}"
                    print(observation)
                elif action == "Lookup":
                    if wiki_page is None:
                        raise ValueError("Cannot lookup without a successful search first")
                    observation = wiki_page.lookup(_input)
                elif action == "Finish":
                    return {"full_logic": prompt, self.output_key: _input}
                else:
                    raise ValueError
                prompt = (
                    prompt
                    + f"\nObservation {i}: "
                    + observation
                    + f"\nThought {i + 1}:"
                )
                i += 1
                stop_seq = f"\nObservation {i}:"
                prefix = ""
            except ActionError:
                prompt = prompt + f"\nAction {i}:"
                stop_seq = f"\nObservation {i}:"
                prefix = f"\nAction {i}:"

    def run(self, question: str) -> str:
        """More user-friendly interface for interfacing with react."""
        return self({self.input_key: question})[self.output_key]
