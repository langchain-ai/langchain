"""Combining documents by mapping a chain over them first, then combining results."""

from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Extra, root_validator

from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.docstore.document import Document


class MapReduceDocumentsChain(BaseCombineDocumentsChain, BaseModel):
    """Combining documents by mapping a chain over them, then combining results."""

    llm_chain: LLMChain
    """Chain to apply to each document individually.."""
    combine_document_chain: BaseCombineDocumentsChain
    """Chain to use to combine results of applying llm_chain to documents."""
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

    def combine_docs(self, docs: List[Document], **kwargs: Any) -> str:
        """Combine by mapping first chain over all, then stuffing into final chain."""
        results = self.llm_chain.apply(
            [{**{self.document_variable_name: d.page_content}, **kwargs} for d in docs]
        )
        question_result_key = self.llm_chain.output_key
        result_docs = [
            Document(page_content=r[question_result_key], metadata=docs[i].metadata)
            for i, r in enumerate(results)
        ]
        return self.combine_document_chain.combine_docs(result_docs, **kwargs)
