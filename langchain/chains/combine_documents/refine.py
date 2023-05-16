"""Combining documents by doing a first pass and then refining on more documents."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from pydantic import Extra, Field, root_validator

from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import Callbacks
from langchain.chains.combine_documents.base import (
    BaseCombineDocumentsChain,
    format_document,
    get_default_document_prompt,
)
from langchain.chains.llm import LLMChain
from langchain.docstore.document import Document
from langchain.prompts.base import BasePromptTemplate
from langchain.prompts.prompt import PromptTemplate


class RefineDocumentsChain(BaseCombineDocumentsChain):
    """Combine documents by doing a first pass and then refining on more documents."""

    initial_llm_chain: LLMChain
    """LLM chain to use on initial document."""
    refine_llm_chain: LLMChain
    """LLM chain to use when refining."""
    document_variable_name: str
    """The variable name in the initial_llm_chain to put the documents in.
    If only one variable in the initial_llm_chain, this doesn't need to be specified.
    """
    initial_response_name: str
    """The variable name to format the initial response in when refining.
    If only two variables are in the refine_llm_chain, this doesn't need to be
    specified.
    """
    document_prompt: BasePromptTemplate = Field(
        default_factory=get_default_document_prompt
    )
    """Prompt to use to format each document."""
    return_intermediate_steps: bool = False
    """Return the results of the refine steps in the output."""

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        question_prompt: BasePromptTemplate,
        refine_prompt: BasePromptTemplate,
        **kwargs: Any,
    ) -> RefineDocumentsChain:
        """Initialize RefineDocumentsChain from an LLM and two prompts.

        Example:
            from langchain.chains.combine_documents import RefineDocumentsChain
            from langchain.chains.summarize.refine_prompts import PROMPT, REFINE_PROMPT
            from langchain.llms import OpenAI

            refine_docs_chain = RefineDocumentsChain.from_llm(
                OpenAI(), PROMPT, REFINE_PROMPT
            )
        """
        initial_chain = LLMChain(llm=llm, prompt=question_prompt)
        refine_chain = LLMChain(llm=llm, prompt=refine_prompt)
        return RefineDocumentsChain(
            initial_llm_chain=initial_chain,
            refine_llm_chain=refine_chain,
            **kwargs,
        )

    @property
    def input_keys(self) -> List[str]:
        """Return input keys."""
        keys = set(
            [self.input_documents_key]
            + self.initial_llm_chain.input_keys
            + self.refine_llm_chain.input_keys
        ).difference([self.document_variable_name, self.initial_response_name])
        return list(keys)

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
        if "return_refine_steps" in values:
            values["return_intermediate_steps"] = values["return_refine_steps"]
            del values["return_refine_steps"]
        return values

    @root_validator(pre=True)
    def get_default_variable_names(cls, values: Dict) -> Dict:
        """Infer and validate sub-chain input variable names, if not provided."""
        initial_inputs = values["initial_llm_chain"].input_keys
        if "document_variable_name" not in values:
            if len(initial_inputs) == 1:
                values["document_variable_name"] = initial_inputs[0]
            else:
                raise ValueError(
                    "document_variable_name must be provided if there are "
                    "multiple initial_llm_chain input_variables"
                )
        else:
            if values["document_variable_name"] not in initial_inputs:
                raise ValueError(
                    f"document_variable_name {values['document_variable_name']} was "
                    f"not found in initial_llm_chain input_keys: {initial_inputs}"
                )
        refine_inputs = values["refine_llm_chain"].input_keys
        if "initial_response_name" not in values:
            doc_input = values["document_variable_name"]
            if len(refine_inputs) == 2:
                init_resp_input = [i for i in refine_inputs if i != doc_input][0]
                values["initial_response_name"] = init_resp_input
            else:
                raise ValueError
        else:
            if values["initial_response_name"] not in refine_inputs:
                raise ValueError(
                    f"initial_response_name {values['initial_response_name']} was not "
                    f"found in refine_llm_chain input_keys: {refine_inputs}"
                )
        return values

    def combine_docs(
        self, docs: List[Document], callbacks: Callbacks = None, **kwargs: Any
    ) -> Tuple[str, dict]:
        """Combine by mapping first chain over all, then stuffing into final chain."""
        inputs = self._construct_initial_inputs(docs, **kwargs)
        res = self.initial_llm_chain.predict(callbacks=callbacks, **inputs)
        refine_steps = [res]
        for doc in docs[1:]:
            inputs = self._construct_refine_inputs(doc, res)
            res = self.refine_llm_chain.predict(callbacks=callbacks, **inputs)
            refine_steps.append(res)
        return self._construct_result(refine_steps, res)

    async def acombine_docs(
        self, docs: List[Document], callbacks: Callbacks = None, **kwargs: Any
    ) -> Tuple[str, dict]:
        """Combine by mapping first chain over all, then stuffing into final chain."""
        inputs = self._construct_initial_inputs(docs, **kwargs)
        res = await self.initial_llm_chain.apredict(callbacks=callbacks, **inputs)
        refine_steps = [res]
        for doc in docs[1:]:
            inputs = self._construct_refine_inputs(doc, res)
            res = await self.refine_llm_chain.apredict(callbacks=callbacks, **inputs)
            refine_steps.append(res)
        return self._construct_result(refine_steps, res)

    def _construct_result(self, refine_steps: List[str], res: str) -> Tuple[str, dict]:
        if self.return_intermediate_steps:
            extra_return_dict = {"intermediate_steps": refine_steps}
        else:
            extra_return_dict = {}
        return res, extra_return_dict

    def _construct_refine_inputs(
        self, doc: Document, res: str, **kwargs: Any
    ) -> Dict[str, Any]:
        return {
            self.document_variable_name: format_document(doc, self.document_prompt),
            self.initial_response_name: res,
            **kwargs,
        }

    def _construct_initial_inputs(
        self, docs: List[Document], **kwargs: Any
    ) -> Dict[str, Any]:
        return {
            self.document_variable_name: format_document(docs[0], self.document_prompt),
            **kwargs,
        }

    @property
    def _chain_type(self) -> str:
        return "refine_documents_chain"
