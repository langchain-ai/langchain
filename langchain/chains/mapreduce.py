"""Map-reduce chain.

Splits up a document, sends the smaller parts to the LLM with one prompt,
then combines the results with another one.
"""
from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, Extra

from langchain.chains.base import Chain
from langchain.chains.combine_documents import CombineDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.docstore.document import Document
from langchain.llms.base import LLM
from langchain.prompts.base import BasePromptTemplate
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
        cls, llm: LLM, prompt: BasePromptTemplate, text_splitter: TextSplitter
    ) -> MapReduceChain:
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

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        # Split the larger text into smaller chunks.
        docs = self.text_splitter.split_text(inputs[self.input_key])

        # Now that we have the chunks, we send them to the LLM and track results.
        #  This is the "map" part.
        input_list = [{self.map_llm.prompt.input_variables[0]: d} for d in docs]
        summary_results = self.map_llm.apply(input_list)
        summaries = [res[self.map_llm.output_key] for res in summary_results]
        summary_docs = [Document(page_content=text) for text in summaries]
        # We then need to combine these individual parts into one.
        # This is the reduce part.
        reduce_chain = CombineDocumentsChain(llm_chain=self.reduce_llm)
        outputs = reduce_chain({reduce_chain.input_key: summary_docs})
        return {self.output_key: outputs[self.output_key]}
