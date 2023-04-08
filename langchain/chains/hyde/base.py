"""Hypothetical Document Embeddings.

https://arxiv.org/abs/2212.10496
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from pydantic import Extra

from langchain.chains.base import Chain
from langchain.chains.hyde.prompts import PROMPT_MAP
from langchain.chains.llm import LLMChain
from langchain.embeddings.base import Embeddings
from langchain.llms.base import BaseLLM


class HypotheticalDocumentEmbedder(Chain, Embeddings):
    """Generate hypothetical document for query, and then embed that.

    Based on https://arxiv.org/abs/2212.10496
    """

    base_embeddings: Embeddings
    llm_chain: LLMChain

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Input keys for Hyde's LLM chain."""
        return self.llm_chain.input_keys

    @property
    def output_keys(self) -> List[str]:
        """Output keys for Hyde's LLM chain."""
        return self.llm_chain.output_keys

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Call the base embeddings."""
        return self.base_embeddings.embed_documents(texts)

    def combine_embeddings(self, embeddings: List[List[float]]) -> List[float]:
        """Combine embeddings into final embeddings."""
        return list(np.array(embeddings).mean(axis=0))

    def embed_query(self, text: str) -> List[float]:
        """Generate a hypothetical document and embedded it."""
        var_name = self.llm_chain.input_keys[0]
        result = self.llm_chain.generate([{var_name: text}])
        documents = [generation.text for generation in result.generations[0]]
        embeddings = self.embed_documents(documents)
        return self.combine_embeddings(embeddings)

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        """Call the internal llm chain."""
        return self.llm_chain._call(inputs)

    @classmethod
    def from_llm(
        cls, llm: BaseLLM, base_embeddings: Embeddings, prompt_key: str
    ) -> HypotheticalDocumentEmbedder:
        """Load and use LLMChain for a specific prompt key."""
        prompt = PROMPT_MAP[prompt_key]
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        return cls(base_embeddings=base_embeddings, llm_chain=llm_chain)

    @property
    def _chain_type(self) -> str:
        return "hyde_chain"
