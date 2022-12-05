"""Combining documents by mapping a chain over them first, then combining results."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from pydantic import BaseModel, Extra, root_validator

from langchain.chains.base import Chain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chains.qa_with_sources.prompt import (
    COMBINE_PROMPT,
    EXAMPLE_PROMPT,
    QUESTION_PROMPT,
)
from langchain.docstore.document import Document
from langchain.llms.base import LLM
from langchain.prompts.base import BasePromptTemplate
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain


class BaseQAWithSourcesChain(BaseCombineDocumentsChain, BaseModel):
    """Question answering with sources over documents."""

    llm_question_chain: LLMChain
    """LLM wrapper to use for asking questions to each document."""
    combine_document_chain: StuffDocumentsChain
    """Chain to use to combine documents."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    def combine_docs(self, docs: List[Document], **kwargs:Any) -> str:
        """Combine by mapping first chain over all, then stuffing into final chain."""
        content_key, query_key = self.llm_question_chain.input_keys
        results = self.llm_question_chain.apply(
            [{**{content_key: d.page_content}, **kwargs} for d in docs]
        )
        question_result_key = self.llm_question_chain.output_key
        result_docs = [
            Document(page_content=r[question_result_key], metadata=docs[i].metadata)
            for i, r in enumerate(results)
        ]
        return self.combine_document_chain.combine_docs(result_docs, kwargs)
