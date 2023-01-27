"""Combining documents by doing a first pass and then refining on more documents."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from pydantic import BaseModel, Extra, Field, root_validator

from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.docstore.document import Document
from langchain.prompts.base import BasePromptTemplate
from langchain.prompts.prompt import PromptTemplate


def _get_default_document_prompt() -> PromptTemplate:
    return PromptTemplate(input_variables=["page_content"], template="{page_content}")


class RefineDocumentsChain(BaseCombineDocumentsChain, BaseModel):
    """Combine documents by doing a first pass and then refining on more documents."""

    initial_llm_chain: LLMChain
    """LLM chain to use on initial document."""
    refine_llm_chain: LLMChain
    """LLM chain to use when refining."""
    document_variable_name: str
    """The variable name in the initial_llm_chain to put the documents in.
    If only one variable in the initial_llm_chain, this need not be provided."""
    initial_response_name: str
    """The variable name to format the initial response in when refining."""
    document_prompt: BasePromptTemplate = Field(
        default_factory=_get_default_document_prompt
    )
    """Prompt to use to format each document."""
    return_intermediate_steps: bool = False
    """Return the results of the refine steps in the output."""

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
    def get_return_intermediate_steps(cls, values: Dict) -> Dict:
        """For backwards compatibility."""
        if "return_refine_steps" in values:
            values["return_intermediate_steps"] = values["return_refine_steps"]
            del values["return_refine_steps"]
        return values

    @root_validator(pre=True)
    def get_default_document_variable_name(cls, values: Dict) -> Dict:
        """Get default document variable name, if not provided."""
        if "document_variable_name" not in values:
            llm_chain_variables = values["initial_llm_chain"].prompt.input_variables
            if len(llm_chain_variables) == 1:
                values["document_variable_name"] = llm_chain_variables[0]
            else:
                raise ValueError(
                    "document_variable_name must be provided if there are "
                    "multiple llm_chain input_variables"
                )
        else:
            llm_chain_variables = values["initial_llm_chain"].prompt.input_variables
            if values["document_variable_name"] not in llm_chain_variables:
                raise ValueError(
                    f"document_variable_name {values['document_variable_name']} was "
                    f"not found in llm_chain input_variables: {llm_chain_variables}"
                )
        return values

    def combine_docs(self, docs: List[Document], **kwargs: Any) -> Tuple[str, dict]:
        """Combine by mapping first chain over all, then stuffing into final chain."""
        base_info = {"page_content": docs[0].page_content}
        base_info.update(docs[0].metadata)
        document_info = {k: base_info[k] for k in self.document_prompt.input_variables}
        base_inputs: dict = {
            self.document_variable_name: self.document_prompt.format(**document_info)
        }
        inputs = {**base_inputs, **kwargs}
        res = self.initial_llm_chain.predict(**inputs)
        refine_steps = [res]
        for doc in docs[1:]:
            base_info = {"page_content": doc.page_content}
            base_info.update(doc.metadata)
            document_info = {
                k: base_info[k] for k in self.document_prompt.input_variables
            }
            base_inputs = {
                self.document_variable_name: self.document_prompt.format(
                    **document_info
                ),
                self.initial_response_name: res,
            }
            inputs = {**base_inputs, **kwargs}
            res = self.refine_llm_chain.predict(**inputs)
            refine_steps.append(res)
        if self.return_intermediate_steps:
            extra_return_dict = {"intermediate_steps": refine_steps}
        else:
            extra_return_dict = {}
        return res, extra_return_dict

    @property
    def _chain_type(self) -> str:
        return "refine_documents_chain"
