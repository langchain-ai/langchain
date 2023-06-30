"""Combining documents by mapping a chain over them first, then combining results."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from pydantic import Extra, root_validator

from langchain.callbacks.manager import Callbacks
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.chains.combine_documents.reduce import ReduceDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.docstore.document import Document


class MapReduceDocumentsChain(BaseCombineDocumentsChain):
    """Combining documents by mapping a chain over them, then combining results.

    We first call `llm_chain` on each document individually, passing in the
    `page_content` and any other kwargs. This is the `map` step.

    We then process the results of that `map` step in a `reduce` step. The `reduce`
    step involves two other chains:

    - combine_document_chain
    - collapse_document_chain

    `combine_document_chain` is ALWAYS provided. This is final chain that is called.
    We pass all previous results to this chain, and the output of this chain is
    returned as a final result.
    """

    llm_chain: LLMChain
    """Chain to apply to each document individually."""
    reduce_document_chain: BaseCombineDocumentsChain
    """Chain to use to reduce the results of applying `llm_chain` to each doc.
    This typically either a ReduceDocumentChain or StuffDocumentChain."""
    document_variable_name: str
    """The variable name in the llm_chain to put the documents in.
    If only one variable in the llm_chain, this need not be provided."""
    return_intermediate_steps: bool = False
    """Return the results of the map steps in the output."""

    @property
    def output_keys(self) -> List[str]:
        """Expect input key.

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
    def get_reduce_chain(cls, values: Dict) -> Dict:
        """For backwards compatibility."""
        if "combine_document_chain" in values:
            if "reduce_chain" in values:
                raise ValueError(
                    "Both `reduce_document_chain` and `combine_document_chain` "
                    "cannot be provided at the same time. `combine_document_chain` "
                    "is deprecated, please only provide `reduce_document_chain`"
                )
            combine_chain = values["combine_document_chain"]
            collapse_chain = values.get("collapse_document_chain")
            reduce_chain = ReduceDocumentsChain(
                combine_document_chain=combine_chain,
                collapse_document_chain=collapse_chain,
            )
            values["reduce_document_chain"] = reduce_chain
            del values["combine_document_chain"]
            if "collapse_document_chain" in values:
                del values["collapse_document_chain"]

        return values

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
    def collapse_document_chain(self) -> BaseCombineDocumentsChain:
        if isinstance(self.reduce_document_chain, ReduceDocumentsChain):
            if self.reduce_document_chain.collapse_document_chain:
                return self.reduce_document_chain.collapse_document_chain
            else:
                return self.reduce_document_chain.combine_document_chain
        else:
            raise ValueError(
                f"`reduce_document_chain` is of type "
                f"{type(self.reduce_document_chain)} so it does not have "
                f"this attribute."
            )

    @property
    def combine_document_chain(self) -> BaseCombineDocumentsChain:
        if isinstance(self.reduce_document_chain, ReduceDocumentsChain):
            return self.reduce_document_chain.combine_document_chain
        else:
            raise ValueError(
                f"`reduce_document_chain` is of type "
                f"{type(self.reduce_document_chain)} so it does not have "
                f"this attribute."
            )

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
        results = self.llm_chain.apply(
            # FYI - this is parallelized and so it is fast.
            [{self.document_variable_name: d.page_content, **kwargs} for d in docs],
            callbacks=callbacks,
        )
        question_result_key = self.llm_chain.output_key
        result_docs = [
            Document(page_content=r[question_result_key], metadata=docs[i].metadata)
            # This uses metadata from the docs, and the textual results from `results`
            for i, r in enumerate(results)
        ]
        result, extra_return_dict = self.reduce_document_chain.combine_docs(
            result_docs, callbacks=callbacks, **kwargs
        )
        if self.return_intermediate_steps:
            _results = [r[self.llm_chain.output_key] for r in results]
            extra_return_dict["intermediate_steps"] = _results
        return result, extra_return_dict

    async def acombine_docs(
        self, docs: List[Document], callbacks: Callbacks = None, **kwargs: Any
    ) -> Tuple[str, dict]:
        """Combine documents in a map reduce manner.

        Combine by mapping first chain over all documents, then reducing the results.
        This reducing can be done recursively if needed (if there are many documents).
        """
        results = await self.llm_chain.aapply(
            # FYI - this is parallelized and so it is fast.
            [{**{self.document_variable_name: d.page_content}, **kwargs} for d in docs],
            callbacks=callbacks,
        )
        question_result_key = self.llm_chain.output_key
        result_docs = [
            Document(page_content=r[question_result_key], metadata=docs[i].metadata)
            # This uses metadata from the docs, and the textual results from `results`
            for i, r in enumerate(results)
        ]
        result, extra_return_dict = await self.reduce_document_chain.acombine_docs(
            result_docs, callbacks=callbacks, **kwargs
        )
        if self.return_intermediate_steps:
            _results = [r[self.llm_chain.output_key] for r in results]
            extra_return_dict["intermediate_steps"] = _results
        return result, extra_return_dict

    @property
    def _chain_type(self) -> str:
        return "map_reduce_documents_chain"
