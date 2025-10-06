"""Hypothetical Document Embeddings.

https://arxiv.org/abs/2212.10496
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import BasePromptTemplate
from langchain_core.runnables import Runnable
from pydantic import ConfigDict

from langchain_classic.chains.base import Chain
from langchain_classic.chains.hyde.prompts import PROMPT_MAP
from langchain_classic.chains.llm import LLMChain

logger = logging.getLogger(__name__)


class HypotheticalDocumentEmbedder(Chain, Embeddings):
    """Generate hypothetical document for query, and then embed that.

    Based on https://arxiv.org/abs/2212.10496
    """

    base_embeddings: Embeddings
    llm_chain: Runnable

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    @property
    def input_keys(self) -> list[str]:
        """Input keys for Hyde's LLM chain."""
        return self.llm_chain.input_schema.model_json_schema()["required"]

    @property
    def output_keys(self) -> list[str]:
        """Output keys for Hyde's LLM chain."""
        if isinstance(self.llm_chain, LLMChain):
            return self.llm_chain.output_keys
        return ["text"]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Call the base embeddings."""
        return self.base_embeddings.embed_documents(texts)

    def combine_embeddings(self, embeddings: list[list[float]]) -> list[float]:
        """Combine embeddings into final embeddings."""
        try:
            import numpy as np

            return list(np.array(embeddings).mean(axis=0))
        except ImportError:
            logger.warning(
                "NumPy not found in the current Python environment. "
                "HypotheticalDocumentEmbedder will use a pure Python implementation "
                "for internal calculations, which may significantly impact "
                "performance, especially for large datasets. For optimal speed and "
                "efficiency, consider installing NumPy: pip install numpy",
            )
            if not embeddings:
                return []
            num_vectors = len(embeddings)
            return [
                sum(dim_values) / num_vectors
                for dim_values in zip(*embeddings, strict=False)
            ]

    def embed_query(self, text: str) -> list[float]:
        """Generate a hypothetical document and embedded it."""
        var_name = self.input_keys[0]
        result = self.llm_chain.invoke({var_name: text})
        if isinstance(self.llm_chain, LLMChain):
            documents = [result[self.output_keys[0]]]
        else:
            documents = [result]
        embeddings = self.embed_documents(documents)
        return self.combine_embeddings(embeddings)

    def _call(
        self,
        inputs: dict[str, Any],
        run_manager: CallbackManagerForChainRun | None = None,
    ) -> dict[str, str]:
        """Call the internal llm chain."""
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        return self.llm_chain.invoke(
            inputs,
            config={"callbacks": _run_manager.get_child()},
        )

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        base_embeddings: Embeddings,
        prompt_key: str | None = None,
        custom_prompt: BasePromptTemplate | None = None,
        **kwargs: Any,
    ) -> HypotheticalDocumentEmbedder:
        """Load and use LLMChain with either a specific prompt key or custom prompt."""
        if custom_prompt is not None:
            prompt = custom_prompt
        elif prompt_key is not None and prompt_key in PROMPT_MAP:
            prompt = PROMPT_MAP[prompt_key]
        else:
            msg = (
                f"Must specify prompt_key if custom_prompt not provided. Should be one "
                f"of {list(PROMPT_MAP.keys())}."
            )
            raise ValueError(msg)

        llm_chain = prompt | llm | StrOutputParser()
        return cls(base_embeddings=base_embeddings, llm_chain=llm_chain, **kwargs)

    @property
    def _chain_type(self) -> str:
        return "hyde_chain"
