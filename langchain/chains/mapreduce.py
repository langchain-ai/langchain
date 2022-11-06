"""Map-reduce chain.

Splits up a document, sends the smaller parts to the LLM with one prompt,
then combines the results with another one.
"""

from typing import Dict, List

from pydantic import BaseModel, Extra

from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.llms.base import LLM
from langchain.prompts.base import BasePrompt
from langchain.text_splitter import TextSplitter


class MapReduceChain(Chain, BaseModel):
    """Map-reduce chain."""

    map_llm: LLMChain
    """LLM wrapper to use for the map step."""
    reduce_llm: LLMChain
    """LLM wrapper to use for the reduce step."""
    text_splitter: TextSplitter
    """Text splitter to use."""
    input_key: str = "input_text"  #: :meta private:
    output_key: str = "output_text"  #: :meta private:

    @classmethod
    def from_params(
        cls, llm: LLM, prompt: BasePrompt, text_splitter: TextSplitter
    ) -> "MapReduceChain":
        """Construct a map-reduce chain that uses the chain for map and reduce."""
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        return cls(map_llm=llm_chain, reduce_llm=llm_chain, text_splitter=text_splitter)

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
        # Split the larger text into smaller chunks.
        docs = self.text_splitter.split_text(
            inputs[self.input_key],
        )
        # Now that we have the chunks, we send them to the LLM and track results.
        #  This is the "map" part.
        summaries = []
        for d in docs:
            inputs = {self.map_llm.prompt.input_variables[0]: d}
            res = self.map_llm.predict(**inputs)
            summaries.append(res)

        # We then need to combine these individual parts into one.
        # This is the reduce part.
        summary_str = "\n".join(summaries)
        inputs = {self.reduce_llm.prompt.input_variables[0]: summary_str}
        output = self.reduce_llm.predict(**inputs)
        return {self.output_key: output}

    def run(self, text: str) -> str:
        """Run the map-reduce logic on the input text."""
        return self({self.input_key: text})[self.output_key]
