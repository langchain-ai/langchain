"""Question answering with sources over documents."""

from __future__ import annotations

import inspect
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.callbacks import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from langchain_core.pydantic_v1 import Extra, root_validator

from langchain.chains import ReduceDocumentsChain
from langchain.chains.base import Chain
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.chains.qa_with_sources.map_reduce_prompt import (
    COMBINE_PROMPT,
    EXAMPLE_PROMPT,
    QUESTION_PROMPT,
)


class BaseQAWithSourcesChain(Chain, ABC):
    """Question answering chain with sources over documents."""

    combine_documents_chain: BaseCombineDocumentsChain
    """Chain to use to combine documents."""
    question_key: str = "question"  #: :meta private:
    input_docs_key: str = "docs"  #: :meta private:
    answer_key: str = "answer"  #: :meta private:
    sources_answer_key: str = "sources"  #: :meta private:
    return_source_documents: bool = False
    """Return the source documents."""

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        document_prompt: BasePromptTemplate = EXAMPLE_PROMPT,
        question_prompt: BasePromptTemplate = QUESTION_PROMPT,
        combine_prompt: BasePromptTemplate = COMBINE_PROMPT,
        **kwargs: Any,
    ) -> BaseQAWithSourcesChain:
        """Construct the chain from an LLM."""
        llm_question_chain = LLMChain(llm=llm, prompt=question_prompt)
        llm_combine_chain = LLMChain(llm=llm, prompt=combine_prompt)
        combine_results_chain = StuffDocumentsChain(
            llm_chain=llm_combine_chain,
            document_prompt=document_prompt,
            document_variable_name="summaries",
        )
        reduce_documents_chain = ReduceDocumentsChain(  # type: ignore[misc]
            combine_documents_chain=combine_results_chain
        )
        combine_documents_chain = MapReduceDocumentsChain(
            llm_chain=llm_question_chain,
            reduce_documents_chain=reduce_documents_chain,
            document_variable_name="context",
        )
        return cls(
            combine_documents_chain=combine_documents_chain,
            **kwargs,
        )

    @classmethod
    def from_chain_type(
        cls,
        llm: BaseLanguageModel,
        chain_type: str = "stuff",
        chain_type_kwargs: Optional[dict] = None,
        **kwargs: Any,
    ) -> BaseQAWithSourcesChain:
        """Load chain from chain type."""
        _chain_kwargs = chain_type_kwargs or {}
        combine_documents_chain = load_qa_with_sources_chain(
            llm, chain_type=chain_type, **_chain_kwargs
        )
        return cls(combine_documents_chain=combine_documents_chain, **kwargs)

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Expect input key.

        :meta private:
        """
        return [self.question_key]

    @property
    def output_keys(self) -> List[str]:
        """Return output key.

        :meta private:
        """
        _output_keys = [self.answer_key, self.sources_answer_key]
        if self.return_source_documents:
            _output_keys = _output_keys + ["source_documents"]
        return _output_keys

    @root_validator(pre=True)
    def validate_naming(cls, values: Dict) -> Dict:
        """Fix backwards compatibility in naming."""
        if "combine_document_chain" in values:
            values["combine_documents_chain"] = values.pop("combine_document_chain")
        return values

    def _split_sources(self, answer: str) -> Tuple[str, str]:
        """Split sources from answer."""
        if re.search(r"SOURCES?:", answer, re.IGNORECASE):
            answer, sources = re.split(
                r"SOURCES?:|QUESTION:\s", answer, flags=re.IGNORECASE
            )[:2]
            sources = re.split(r"\n", sources)[0].strip()
        else:
            sources = ""
        return answer, sources

    @abstractmethod
    def _get_docs(
        self,
        inputs: Dict[str, Any],
        *,
        run_manager: CallbackManagerForChainRun,
    ) -> List[Document]:
        """Get docs to run questioning over."""

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        accepts_run_manager = (
            "run_manager" in inspect.signature(self._get_docs).parameters
        )
        if accepts_run_manager:
            docs = self._get_docs(inputs, run_manager=_run_manager)
        else:
            docs = self._get_docs(inputs)  # type: ignore[call-arg]

        answer = self.combine_documents_chain.run(
            input_documents=docs, callbacks=_run_manager.get_child(), **inputs
        )
        answer, sources = self._split_sources(answer)
        result: Dict[str, Any] = {
            self.answer_key: answer,
            self.sources_answer_key: sources,
        }
        if self.return_source_documents:
            result["source_documents"] = docs
        return result

    @abstractmethod
    async def _aget_docs(
        self,
        inputs: Dict[str, Any],
        *,
        run_manager: AsyncCallbackManagerForChainRun,
    ) -> List[Document]:
        """Get docs to run questioning over."""

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        _run_manager = run_manager or AsyncCallbackManagerForChainRun.get_noop_manager()
        accepts_run_manager = (
            "run_manager" in inspect.signature(self._aget_docs).parameters
        )
        if accepts_run_manager:
            docs = await self._aget_docs(inputs, run_manager=_run_manager)
        else:
            docs = await self._aget_docs(inputs)  # type: ignore[call-arg]
        answer = await self.combine_documents_chain.arun(
            input_documents=docs, callbacks=_run_manager.get_child(), **inputs
        )
        answer, sources = self._split_sources(answer)
        result: Dict[str, Any] = {
            self.answer_key: answer,
            self.sources_answer_key: sources,
        }
        if self.return_source_documents:
            result["source_documents"] = docs
        return result


class QAWithSourcesChain(BaseQAWithSourcesChain):
    """Question answering with sources over documents."""

    input_docs_key: str = "docs"  #: :meta private:

    @property
    def input_keys(self) -> List[str]:
        """Expect input key.

        :meta private:
        """
        return [self.input_docs_key, self.question_key]

    def _get_docs(
        self,
        inputs: Dict[str, Any],
        *,
        run_manager: CallbackManagerForChainRun,
    ) -> List[Document]:
        """Get docs to run questioning over."""
        return inputs.pop(self.input_docs_key)

    async def _aget_docs(
        self,
        inputs: Dict[str, Any],
        *,
        run_manager: AsyncCallbackManagerForChainRun,
    ) -> List[Document]:
        """Get docs to run questioning over."""
        return inputs.pop(self.input_docs_key)

    @property
    def _chain_type(self) -> str:
        return "qa_with_sources_chain"
