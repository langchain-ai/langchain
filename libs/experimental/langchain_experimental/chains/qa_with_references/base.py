"""Question answering with references over documents."""

import inspect
import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple, cast

from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.chains.base import Chain
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain
from langchain.chains.combine_documents.reduce import ReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.docstore.document import Document
from langchain.pydantic_v1 import Extra
from langchain.schema import BaseOutputParser, BasePromptTemplate, OutputParserException

from .loading import (
    load_qa_with_references_chain,
)
from .map_reduce_prompts import (
    COMBINE_PROMPT,
    EXAMPLE_PROMPT,
    QUESTION_PROMPT,
)

logger = logging.getLogger(__name__)


# TOTRY: Add in VectorStoreIndexWrapper.query_with_references()
# TOTRY: Add in VectorStoreIndexWrapper.query_with_references_and_verbatims()
class BaseQAWithReferencesChain(Chain, ABC):
    combine_documents_chain: BaseCombineDocumentsChain

    """Chain to use to combine documents."""
    question_key: str = "question"  #: :meta private:
    input_docs_key: str = "docs"  #: :meta private:
    answer_key: str = "answer"  #: :meta private:
    source_documents_key: str = "source_documents"  #: :meta private:
    chain_type: str  #: :meta private:
    output_parser: Optional[BaseOutputParser] = None

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        document_prompt: BasePromptTemplate = EXAMPLE_PROMPT,
        question_prompt: BasePromptTemplate = QUESTION_PROMPT,
        combine_prompt: BasePromptTemplate = COMBINE_PROMPT,
        **kwargs: Any,
    ) -> "BaseQAWithReferencesChain":
        """Construct the chain from an LLM."""
        llm_question_chain = LLMChain(llm=llm, prompt=question_prompt)
        llm_combine_chain = LLMChain(llm=llm, prompt=combine_prompt)
        combine_results_chain = StuffDocumentsChain(
            llm_chain=llm_combine_chain,
            document_prompt=document_prompt,
            document_variable_name="summaries",
        )
        reduce_documents_chain = ReduceDocumentsChain(
            combine_documents_chain=combine_results_chain
        )
        combine_document_chain = MapReduceDocumentsChain(
            llm_chain=llm_question_chain,
            reduce_documents_chain=reduce_documents_chain,
            document_variable_name="context",
            return_intermediate_steps=True,
        )
        return cls(
            combine_documents_chain=combine_document_chain,
            **kwargs,
        )

    @classmethod
    def from_chain_type(
        cls,
        llm: BaseLanguageModel,
        chain_type: str = "stuff",
        chain_type_kwargs: Optional[dict] = None,
        **kwargs: Any,
    ) -> "BaseQAWithReferencesChain":
        """Load chain from chain type."""
        _chain_kwargs = chain_type_kwargs or {}
        combine_document_chain = load_qa_with_references_chain(
            llm,
            chain_type=chain_type,
            **_chain_kwargs,
        )
        if chain_type == "map_rerank" and "output_parser" not in kwargs:
            from .map_rerank_prompts import rerank_reference_parser

            kwargs["output_parser"] = rerank_reference_parser

        return cls(
            combine_documents_chain=combine_document_chain,
            chain_type=chain_type,
            **kwargs,
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
        return [self.question_key]

    @property
    def output_keys(self) -> List[str]:
        """Return output key.

        :meta private:
        """
        _output_keys = [self.answer_key, self.source_documents_key]
        return _output_keys

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

    def _process_reference(
        self, answers: Dict[str, Any], docs: List[Document], references: Any
    ) -> Set[int]:
        # With the map_rerank mode, use extra parameter for map_rerank to identify
        # the corresponding document.
        if "_idx" in answers:
            references.documents_ids.add(answers["_idx"])

        documents_idx = set()
        for str_doc_id in references.documents_ids:
            m = re.match(r"(?:_idx_)?(\d+)", str_doc_id.strip())
            if m:
                documents_idx.add(int(m[1]))
            else:
                pass
        return documents_idx

    def _process_results(
        self,
        answers: Dict[str, Any],
        docs: List[Document],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Tuple[str, Set[int]]:
        idx = set()
        answer = answers[self.combine_documents_chain.output_key]
        # At this time (version 0.0.273), the output_parser is not
        # automatically called.
        # We must extract this parser to analyse the response.
        parser: Optional[BaseOutputParser] = self.output_parser
        llm_chain: Optional[LLMChain] = None
        if not parser:
            if self.chain_type == "map_rerank":
                # self.combine_documents_chain.llm_chain.prompt.output_parser
                assert self.output_parser
            elif self.chain_type == "map_reduce":
                llm_chain = cast(
                    Any, self.combine_documents_chain
                ).collapse_document_chain.llm_chain
            elif self.chain_type == "refine":
                llm_chain = cast(Any, self.combine_documents_chain).refine_llm_chain
            else:
                # Simple chain
                llm_chain = cast(Any, self.combine_documents_chain).llm_chain
        if llm_chain:
            parser = llm_chain.prompt.output_parser
        assert parser

        try:
            references = parser.parse(answer)
            answer = references.response

            idx = self._process_reference(answers, docs, references)

            for doc in docs:
                del doc.metadata["_idx"]
            return answer, idx
        except (OutputParserException, ValueError) as e:
            # Probably that the answer has been cut off.
            raise OutputParserException(
                "The response is probably cut off. Change the `max_tokens` parameter "
                "or reduce the temperature.\n" + str(e)
            ).with_traceback(e.__traceback__)
        # except Exception as e:
        #     if run_manager:
        #         run_manager.on_chain_error(e)
        #     raise e

    @abstractmethod
    async def _aget_docs(
        self,
        inputs: Dict[str, Any],
        *,
        run_manager: AsyncCallbackManagerForChainRun,
    ) -> List[Document]:
        """Get docs to run questioning over."""

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        accepts_run_manager = (
            "run_manager" in inspect.signature(self._get_docs).parameters
        )
        if accepts_run_manager:
            docs = self._get_docs(inputs, run_manager=_run_manager)
        else:
            docs = self._get_docs(inputs)  # type: ignore[call-arg]

        # Inject position in the list
        # To avoid confusion with other id, add a prefix
        for idx, doc in enumerate(docs):
            doc.metadata["_idx"] = f"_idx_{idx}"

        answers = self.combine_documents_chain(
            {
                self.combine_documents_chain.input_key: docs,
                self.question_key: inputs[self.question_key],
                **inputs,
            },
            callbacks=_run_manager.get_child(),
        )
        answer, all_idx = self._process_results(answers, docs)
        selected_docs = [docs[idx] for idx in all_idx if idx < len(docs)]

        return {
            self.answer_key: answer,
            self.source_documents_key: selected_docs,
        }

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

        # Inject position in the list
        for idx, doc in enumerate(docs):
            doc.metadata["_idx"] = idx

        await self.combine_documents_chain.arun(
            input_documents=docs, callbacks=_run_manager.get_child(), **inputs
        )
        answers = await self.combine_documents_chain.acall(
            {
                self.combine_documents_chain.input_key: docs,
                self.question_key: inputs[self.question_key],
                **inputs,
            },
            callbacks=_run_manager.get_child(),
        )
        answer, all_idx = self._process_results(answers, docs)
        selected_docs = [docs[idx] for idx in all_idx if idx < len(docs)]

        return {
            self.answer_key: answer,
            self.source_documents_key: selected_docs,
        }


class QAWithReferencesChain(BaseQAWithReferencesChain):
    """
    This chain extracts the information from the documents that was used to answer the
    question. The output `source_documents` contains only the documents that were used,
    and for each one.

    The result["source_documents"] returns only the list of documents used.
    Then it is possible to find the page of a PDF, or the chapter of a markdown.
    """

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
        return "qa_with_references_chain"
