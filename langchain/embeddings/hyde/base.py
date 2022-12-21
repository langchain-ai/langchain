"""Hypothetical Document Embeddings.

http://boston.lti.cs.cmu.edu/luyug/HyDE/HyDE.pdf
"""
from __future__ import annotations

from typing import List

from pydantic import BaseModel, Extra

from langchain.chains.llm import LLMChain
from langchain.embeddings.base import Embeddings
from langchain.embeddings.hyde.prompts import PROMPT_MAP
from langchain.llms.base import LLM


class HypotheticalDocumentEmbedder(Embeddings, BaseModel):
    """Generate hypothetical document for query, and then embed that.

    Based on http://boston.lti.cs.cmu.edu/luyug/HyDE/HyDE.pdf
    """

    base_embeddings: Embeddings
    llm_chain: LLMChain

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Call the base embeddings."""
        return self.base_embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        """Generate a hypothetical document and embedded it."""
        document = self.llm_chain.run(text)
        embeddings = self.embed_documents([document])
        return embeddings[0]

    @classmethod
    def from_llm(
        cls, llm: LLM, base_embeddings: Embeddings, prompt_key: str
    ) -> HypotheticalDocumentEmbedder:
        """Load and use LLMChain for a specific prompt key."""
        prompt = PROMPT_MAP[prompt_key]
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        return cls(base_embeddings=base_embeddings, llm_chain=llm_chain)
