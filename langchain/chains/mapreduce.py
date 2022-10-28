"""Map-reduce chain."""

from typing import Dict, List

from pydantic import BaseModel, Extra

from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.llms.base import LLM
from langchain.prompt import Prompt


class MapReduceChain(Chain, BaseModel):

    llm: LLM
    """LLM wrapper to use."""
    map_prompt: Prompt
    reduce_prompt: Prompt
    reduce_descriptor: str
    chunk_size: int = 4000
    chunk_overlap: int = 200
    separator: str = "\n\n"
    input_key: str = "input_text"  #: :meta private:
    output_key: str = "output_text"  #: :meta private:

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
        """Return output key.

        :meta private:
        """
        return [self.output_key]

    def _run(self, inputs: Dict[str, str]) -> Dict[str, str]:
        map_llm = LLMChain(llm=self.llm, prompt=self.map_prompt)
        reduce_llm = LLMChain(llm=self.llm, prompt=self.reduce_prompt)
        # First we naively split the large input into a bunch of smaller ones.
        splits = inputs[self.input_key].split(self.separator)
        # We now want to combine these smaller pieces into medium size
        # chunks to send to the LLM. This is the "map" part.
        docs = []
        current_doc: List[str] = []
        total = 0
        for d in splits:
            if total > self.chunk_size:
                docs.append(self.separator.join(current_doc))
                while total > self.chunk_overlap:
                    total -= len(current_doc[0])
                    current_doc = current_doc[1:]
            current_doc.append(d)
            total += len(d)
        docs.append(self.separator.join(current_doc))
        # Now that we have the chunks, we send them to the LLM and track results.
        summaries = []
        for d in docs:
            inputs = {self.map_prompt.input_variables[0]: d}
            res = map_llm.predict(**inputs)
            summaries.append(res)

        # We then need to combine these individual parts into one.
        # This is the reduce part.
        summary_str = "\n\n".join(
            [f"{self.reduce_descriptor} {i}: {s}" for i, s in enumerate(summaries)]
        )
        inputs = {self.reduce_prompt.input_variables[0]: summary_str}
        output = reduce_llm.predict(**inputs)
        return {self.output_key: output}

    def run(self, text: str) -> str:
        return self({self.input_key: text})[self.output_key]
