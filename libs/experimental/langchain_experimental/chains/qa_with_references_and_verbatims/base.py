"""Question answering with references and verbatims over documents."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set, cast

from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.chains import ReduceDocumentsChain
from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.docstore.document import Document
from langchain.schema import BasePromptTemplate, OutputParserException

from ..qa_with_references.base import BaseQAWithReferencesChain
from .loading import (
    load_qa_with_references_chain,
)
from .map_reduce_prompts import (
    COMBINE_PROMPT,
    EXAMPLE_PROMPT,
    QUESTION_PROMPT,
)
from .verbatims import Verbatims, verbatims_parser

logger = logging.getLogger(__name__)


class BaseQAWithReferencesAndVerbatimsChain(BaseQAWithReferencesChain):
    original_verbatim: bool = False
    """ Set to true to extract the verbatim in the document (via regex) """

    def _process_reference(
        self, answers: Dict[str, Any], docs: List[Document], references: Any
    ) -> Set[int]:
        idx = set()
        try:
            verbatims = cast(Verbatims, references)
            # With the map_rerank mode, use extra parameter for map_rerank to identify
            # the corresponding document.
            if "_idx" in answers:
                # Use extra parameter for map_rerank to identify the corresponding
                # document.
                # Fix the ids of the selected document.
                verbatims.documents[0].ids = [answers["_idx"]]

            # Inject verbatims and get idx
            for ref_doc in verbatims.documents:
                for str_doc_id in ref_doc.ids:
                    m = re.match(r"_idx_(\d+)", str_doc_id)
                    if not m:
                        logger.debug(f"Detected invalid ids '{str_doc_id}'")
                        continue
                    doc_id = int(m[1])
                    if 0 <= doc_id < len(docs):  # Guard
                        if ref_doc.verbatims:
                            # Search the real verbatim if possible
                            if self.original_verbatim:
                                original_verbatim = ref_doc.original_verbatims(
                                    docs[doc_id].page_content
                                )
                            else:
                                original_verbatim = ref_doc.verbatims
                            if original_verbatim:
                                idx.add(doc_id)
                                docs[doc_id].metadata["verbatims"] = original_verbatim
                            elif "verbatims" not in docs[doc_id].metadata:
                                logger.debug(
                                    f"Verbatim not confirmed in original document\n"
                                    f"{ref_doc.verbatims}"
                                )
                                # docs[doc_id].metadata["verbatims"] =ref_doc.verbatims
            return idx
        except OutputParserException as e:
            logger.debug(f"Exception during parsing: {e}")
            # return idx
            raise

    @property
    def _chain_type(self) -> str:
        return "qa_with_references_and_verbatims_chain"

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        document_prompt: BasePromptTemplate = EXAMPLE_PROMPT,
        question_prompt: BasePromptTemplate = QUESTION_PROMPT,
        combine_prompt: BasePromptTemplate = COMBINE_PROMPT,
        **kwargs: Any,
    ) -> BaseQAWithReferencesChain:
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
    ) -> BaseQAWithReferencesChain:
        """Load chain from chain type."""
        _chain_kwargs = chain_type_kwargs or {}
        combine_document_chain = load_qa_with_references_chain(
            llm,
            chain_type=chain_type,
            **_chain_kwargs,
        )
        if chain_type == "map_rerank" and "output_parser" not in kwargs:
            kwargs["output_parser"] = verbatims_parser

        return cls(
            combine_documents_chain=combine_document_chain,
            chain_type=chain_type,
            **kwargs,
        )


class QAWithReferencesAndVerbatimsChain(BaseQAWithReferencesAndVerbatimsChain):
    """
    Question answering with references and verbatims over documents.

    This chain extracts the information from the documents that was used to answer the
    question. The output `source_documents` contains only the documents that were used,
    and for each one, only the text fragments that were used to answer are included.
    If possible, the list of text fragments that justify the answer is added to
    `metadata['verbatims']` for each document.
    Then it is possible to find the page of a PDF, or the chapter of a markdown.

    The incogenent of this chain, and that this consumes output tokens.

    A sample result of usage with Wikipedia, may be:
    ```
    For the question "what can you say about ukraine?",
    to answer "Ukraine has an illegal and unprovoked invasion inside its territory,
    and its citizens show courage and are fighting against it. It is believed that
    the capital city of Kyiv, which is home to 2.8 million people, is a target.",
    the LLM use:
    Source https://www.defense.gov/News/Transcripts/...
    -  "when it comes to conducting their illegal and unprovoked invasion inside
    Ukraine."
    Source https://www.whitehouse.gov/briefing-room/...
    - "to the fearless and skilled Ukrainian fighters who are standing in the
    breach"
    - "You got to admit, you have -- must be amazed at the courage of this country"
    Source https://www.whitehouse.gov/briefing-room/...
    -  "believe that they will target Ukraine's capital, Kyiv, a city of 2.8 million
    innocent people."
    Source https://www.whitehouse.gov/briefing-room/...
    -  "believe that they will target Ukraine's capital, Kyiv, a city of 2.8 million
    innocent people."
    ```
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
        return "qa_with_references_and_verbatim_chain"
