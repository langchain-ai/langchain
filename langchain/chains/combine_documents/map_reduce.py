"""Combining documents by mapping a chain over them first, then combining results."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Extra, root_validator

from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.docstore.document import Document


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
    combine_document_func: Callable,
    **kwargs: Any,
) -> Document:
    result = combine_document_func(docs, **kwargs)
    combined_metadata = {k: str(v) for k, v in docs[0].metadata.items()}
    for doc in docs[1:]:
        for k, v in doc.metadata.items():
            if k in combined_metadata:
                combined_metadata[k] += f", {v}"
            else:
                combined_metadata[k] = str(v)
    return Document(page_content=result, metadata=combined_metadata)


class MapReduceDocumentsChain(BaseCombineDocumentsChain, BaseModel):
    """Combining documents by mapping a chain over them, then combining results."""

    llm_chain: LLMChain
    """Chain to apply to each document individually."""
    combine_document_chain: BaseCombineDocumentsChain
    """Chain to use to combine results of applying llm_chain to documents."""
    collapse_document_chain: Optional[BaseCombineDocumentsChain] = None
    """Chain to use to collapse intermediary results if needed.
    If None, will use the combine_document_chain."""
    document_variable_name: str
    """The variable name in the llm_chain to put the documents in.
    If only one variable in the llm_chain, this need not be provided."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

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

    def combine_docs(
        self, docs: List[Document], token_max: int = 3000, **kwargs: Any
    ) -> str:
        """Combine documents in a map reduce manner.

        Combine by mapping first chain over all documents, then reducing the results.
        This reducing can be done recursively if needed (if there are many documents).
        """
        results = self.llm_chain.apply(
            # FYI - this is parallelized and so it is fast.
            [{**{self.document_variable_name: d.page_content}, **kwargs} for d in docs]
        )
        question_result_key = self.llm_chain.output_key
        result_docs = [
            Document(page_content=r[question_result_key], metadata=docs[i].metadata)
            # This uses metadata from the docs, and the textual results from `results`
            for i, r in enumerate(results)
        ]
        length_func = self.combine_document_chain.prompt_length
        num_tokens = length_func(result_docs, **kwargs)
        while num_tokens is not None and num_tokens > token_max:
            new_result_doc_list = _split_list_of_docs(
                result_docs, length_func, token_max, **kwargs
            )
            result_docs = []
            for docs in new_result_doc_list:
                new_doc = _collapse_docs(
                    docs, self._collapse_chain.combine_docs, **kwargs
                )
                result_docs.append(new_doc)
            num_tokens = self.combine_document_chain.prompt_length(
                result_docs, **kwargs
            )
        output = self.combine_document_chain.combine_docs(result_docs, **kwargs)
        return output
