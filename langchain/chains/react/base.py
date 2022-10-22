"""Chain that implements the ReAct Methodology."""
from typing import Any, Dict, List

from pydantic import BaseModel, Extra

from langchain.chains.base import Chain
from langchain.llms.base import LLM
from langchain.chains.llm import LLMChain
from langchain.chains.react.prompt import PROMPT

import wikipedia


def wikipedia_search(search):
    try:
        lookup = wikipedia.summary(search).split("\n")[0]
    except wikipedia.PageError:
        lookup = f"Could not find [{search}]. Similar: {wikipedia.search(search)}"

    except wikipedia.DisambiguationError:
        lookup = f"Could not find [{search}]. Similar: {wikipedia.search(search)}"
    return lookup

def wikipedia_lookup(search, lookup):
    page_content = wikipedia.page(search).content
    lookups = [p for p in page_content.split('\n') if lookup.lower() in p.lower()]
    if len(lookups) == 0:
        return "No Results"
    return f"(Result {1}/{len(lookups)}) {lookups[0]}"

import re
class ActionError(Exception):
    pass
def extract_action(text, i):
    action_block = text.split('\n')[-1]
    action_prefix = f"Action {i}: "
    if not action_block.startswith(action_prefix):
        raise ActionError
    action_str = action_block[len(action_prefix):]
    re_matches = re.search('(.*?)\[(.*?)\]', action_str)
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
        search_term = None
        while True:
            ret_text = llm_chain.predict(input=prompt, stop=[stop_seq])
            prompt += ret_text
            try:
                action, _input = extract_action(prefix + ret_text, i)
                print(action, _input)
                if action == "Search":
                    observation = wikipedia_search(_input)
                    print(observation)
                    search_term = _input
                elif action == "Lookup":
                    if search_term is None:
                        raise ValueError
                    observation = wikipedia_lookup(search_term, _input)
                elif action == "Finish":
                    return {"full_logic": prompt, self.output_key: _input}
                else:
                    raise ValueError
                prompt = prompt + f"\nObservation {i}: " + observation + f"\nThought {i + 1}:"
                i += 1
                stop_seq = f"\nObservation {i}:"
            except ActionError:
                prompt = prompt + f"\nAction {i}:"
                stop_seq = f"\nObservation {i}:"
                prefix = f"\nAction {i}:"

    def run(self, question: str) -> str:
        """More user-friendly interface for interfacing with react."""
        return self({self.input_key: question})[self.output_key]
