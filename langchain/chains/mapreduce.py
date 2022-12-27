"""Map-reduce chain.

Splits up a document, sends the smaller parts to the LLM with one prompt,
then combines the results with another one.
"""
from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, Extra

from langchain.chains.base import Chain
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.docstore.document import Document
from langchain.llms.base import BaseLLM
from langchain.prompts.base import BasePromptTemplate
from langchain.text_splitter import TextSplitter


class MapReduceChain(Chain, BaseModel):
    """Map-reduce chain."""

    combine_documents_chain: BaseCombineDocumentsChain
    """Chain to use to combine documents."""
    text_splitter: TextSplitter
    """Text splitter to use."""
    input_key: str = "input_text"  #: :meta private:
    output_key: str = "output_text"  #: :meta private:

    @classmethod
    def from_params(
        cls, llm: BaseLLM, prompt: BasePromptTemplate, text_splitter: TextSplitter
    ) -> MapReduceChain:
        """Construct a map-reduce chain that uses the chain for map and reduce."""
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        reduce_chain = StuffDocumentsChain(llm_chain=llm_chain)
        combine_documents_chain = MapReduceDocumentsChain(
            llm_chain=llm_chain, combine_document_chain=reduce_chain
        )
        return cls(
            combine_documents_chain=combine_documents_chain, text_splitter=text_splitter
        )

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
        texts = self.text_splitter.split_text(inputs[self.input_key])
        docs = [Document(page_content=text) for text in texts]
        outputs, _ = self.combine_documents_chain.combine_docs(docs)
        return {self.output_key: outputs}
