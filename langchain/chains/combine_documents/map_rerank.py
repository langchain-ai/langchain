"""Combining documents by mapping a chain over them first, then reranking results."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Extra, root_validator

from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.docstore.document import Document



class MapRerankDocumentsChain(BaseCombineDocumentsChain, BaseModel):
    """Combining documents by mapping a chain over them, then reranking results."""

    llm_chain: LLMChain
    """Chain to apply to each document individually."""
    document_variable_name: str
    """The variable name in the llm_chain to put the documents in.
    If only one variable in the llm_chain, this need not be provided."""
    rank_key: str
    """Key in output of llm_chain to rank on."""
    metadata_keys: Optional[List[str]] = None

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

    def combine_docs(
        self, docs: List[Document], **kwargs: Any
    ) -> str:
        """Combine documents in a map rerank manner.

        Combine by mapping first chain over all documents, then reranking the results.
        """
        results = self.llm_chain.apply_and_parse(
            # FYI - this is parallelized and so it is fast.
            [{**{self.document_variable_name: d.page_content}, **kwargs} for d in docs]
        )
        sorted_res = sorted(zip(results, docs), key=lambda x: -x[0][self.rank_key])
        output, document = sorted_res[0]
        if self.metadata_keys is not None:
            for key in self.metadata_keys:
                output[key] = document.metadata[key]
        return output
