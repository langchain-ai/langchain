"""Combining documents by mapping a chain over them first, then combining results."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

from mypy_extensions import KwArg
from pydantic import Extra, Field, root_validator

from langchain import BasePromptTemplate
from langchain.callbacks.manager import Callbacks
from langchain.chains.combine_documents.base import (
    BaseCombineDocumentsChain,
    format_document,
    get_default_document_prompt,
)
from langchain.chains.llm import LLMChain
from langchain.docstore.document import Document


class CombineDocsProtocol(Protocol):
    """Interface for the combine_docs method."""

    def __call__(self, docs: List[Document], **kwargs: Any) -> str:
        """Interface for the combine_docs method."""


def _split_list_of_docs(
    docs: List[Document], length_func: Callable, token_max: int, **kwargs: Any
) -> List[List[Document]]:
    new_result_doc_list = []
    _sub_result_docs = []
    for doc in docs:
        _sub_result_docs.append(doc)
        _num_tokens = length_func(_sub_result_docs, **kwargs)
        if _num_tokens > token_max:
            if len(_sub_result_docs) == 1:
                raise ValueError(
                    "A single document was longer than the context length,"
                    " we cannot handle this."
                )
            if len(_sub_result_docs) == 2:
                raise ValueError(
                    "A single document was so long it could not be combined "
                    "with another document, we cannot handle this."
                )
            new_result_doc_list.append(_sub_result_docs[:-1])
            _sub_result_docs = _sub_result_docs[-1:]
    new_result_doc_list.append(_sub_result_docs)
    return new_result_doc_list


def _collapse_docs(
    docs: List[Document],
    combine_docs_result: str,
) -> Document:
    combined_metadata = {k: str(v) for k, v in docs[0].metadata.items()}
    for doc in docs[1:]:
        for k, v in doc.metadata.items():
            if k in combined_metadata:
                combined_metadata[k] += f", {v}"
            else:
                combined_metadata[k] = str(v)
    return Document(page_content=combine_docs_result, metadata=combined_metadata)


class MapReduceDocumentsChain(BaseCombineDocumentsChain):
    """Combining documents by mapping a chain over them, then combining results."""

    llm_chain: LLMChain
    """Chain to apply to each document individually."""
    document_prompt: BasePromptTemplate = Field(
        default_factory=get_default_document_prompt
    )
    """Prompt to use to format each document."""
    combine_document_chain: BaseCombineDocumentsChain
    """Chain to use to combine results of applying llm_chain to documents."""
    collapse_document_chain: Optional[BaseCombineDocumentsChain] = None
    """Chain to use to collapse intermediary results if needed.
    If None, will use the combine_document_chain."""
    document_variable_name: str
    """The variable name in the llm_chain to put the documents in.
    If only one variable in the llm_chain, this need not be provided."""
    return_intermediate_steps: bool = False
    """Return the results of the map steps in the output."""

    @property
    def input_keys(self) -> List[str]:
        """Return input keys."""
        all_keys = set(
            [self.input_documents_key, "token_max"]
            + self.llm_chain.input_keys
            + self._collapse_chain.input_keys
        )
        internal_keys = [
            self.document_variable_name,
            self.combine_document_chain.input_documents_key,
            self._collapse_chain.input_documents_key,
        ]
        return list(set(all_keys).difference(internal_keys))

    @property
    def output_keys(self) -> List[str]:
        """Return output keys.

        :meta private:
        """
        _output_keys = super().output_keys
        if self.return_intermediate_steps:
            _output_keys = _output_keys + ["intermediate_steps"]
        return _output_keys

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @root_validator(pre=True)
    def get_return_intermediate_steps(cls, values: Dict) -> Dict:
        """For backwards compatibility."""
        if "return_map_steps" in values:
            values["return_intermediate_steps"] = values["return_map_steps"]
            del values["return_map_steps"]
        return values

    @root_validator(pre=True)
    def get_default_document_variable_name(cls, values: Dict) -> Dict:
        """Get default document variable name, if not provided."""
        if "document_variable_name" not in values:
            llm_chain_variables = values["llm_chain"].prompt.input_variables
            if len(llm_chain_variables) == 1:
                values["document_variable_name"] = llm_chain_variables[0]
            else:
                raise ValueError(
                    "document_variable_name must be provided if there are "
                    "multiple llm_chain input_variables"
                )
        else:
            llm_chain_variables = values["llm_chain"].prompt.input_variables
            if values["document_variable_name"] not in llm_chain_variables:
                raise ValueError(
                    f"document_variable_name {values['document_variable_name']} was "
                    f"not found in llm_chain input_variables: {llm_chain_variables}"
                )
        return values

    @property
    def _collapse_chain(self) -> BaseCombineDocumentsChain:
        if self.collapse_document_chain is not None:
            return self.collapse_document_chain
        else:
            return self.combine_document_chain

    def _get_llm_chain_inputs(self, docs: List[Document], **kwargs: Any) -> List[dict]:
        # Format each document according to the prompt
        doc_strings = [format_document(doc, self.document_prompt) for doc in docs]
        # Join the documents together to put them in the prompt.
        _kwargs = {k: v for k, v in kwargs.items() if k in self.llm_chain.input_keys}
        return [{self.document_variable_name: _doc, **_kwargs} for _doc in doc_strings]

    def combine_docs(
        self,
        docs: List[Document],
        token_max: int = 3000,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Tuple[str, dict]:
        """Combine documents in a map reduce manner.

        Combine by mapping first chain over all documents, then reducing the results.
        This reducing can be done recursively if needed (if there are many documents).
        """
        inputs = self._get_llm_chain_inputs(docs, **kwargs)
        # FYI - this is parallelized and so it is fast.
        results = self.llm_chain.apply(inputs, callbacks=callbacks)
        return self._process_results(
            results, docs, token_max, callbacks=callbacks, **kwargs
        )

    async def acombine_docs(
        self,
        docs: List[Document],
        token_max: int = 3000,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Tuple[str, dict]:
        """Combine documents in a map reduce manner.

        Combine by mapping first chain over all documents, then reducing the results.
        This reducing can be done recursively if needed (if there are many documents).
        """
        inputs = self._get_llm_chain_inputs(docs, **kwargs)
        # FYI - this is parallelized and so it is fast.
        results = await self.llm_chain.aapply(inputs, callbacks=callbacks)
        return self._process_results(
            results, docs, token_max, callbacks=callbacks, **kwargs
        )

    @property
    def _length_func(self) -> Callable[[List[Document], KwArg(Any)], Optional[int]]:
        return self.combine_document_chain.prompt_length  # type: ignore

    def _process_results(
        self,
        results: List[Dict],
        docs: List[Document],
        token_max: int,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Tuple[str, dict]:
        question_result_key = self.llm_chain.output_key
        result_docs = [
            Document(page_content=r[question_result_key], metadata=docs[i].metadata)
            # This uses metadata from the docs, and the textual results from `results`
            for i, r in enumerate(results)
        ]
        num_tokens = self._length_func(result_docs, **kwargs)

        while num_tokens is not None and num_tokens > token_max:
            new_result_doc_list = _split_list_of_docs(
                result_docs, self._length_func, token_max, **kwargs
            )
            result_docs = []
            for docs in new_result_doc_list:
                inputs = {self._collapse_chain.input_documents_key: docs, **kwargs}
                result = self._collapse_chain.run(callbacks=callbacks, **inputs)
                result_docs.append(_collapse_docs(docs, result))
            num_tokens = self._length_func(result_docs, **kwargs)
        if self.return_intermediate_steps:
            _results = [r[self.llm_chain.output_key] for r in results]
            extra_return_dict = {"intermediate_steps": _results}
        else:
            extra_return_dict = {}
        inputs = {
            self.combine_document_chain.input_documents_key: result_docs,
            **kwargs,
        }
        output = self.combine_document_chain.run(callbacks=callbacks, **inputs)
        return output, extra_return_dict

    @property
    def _chain_type(self) -> str:
        return "map_reduce_documents_chain"
