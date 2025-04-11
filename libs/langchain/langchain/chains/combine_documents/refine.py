"""Combine documents by doing a first pass and then refining on more documents."""

from __future__ import annotations

from typing import Any

from langchain_core._api import deprecated
from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document
from langchain_core.prompts import BasePromptTemplate, format_document
from langchain_core.prompts.prompt import PromptTemplate
from pydantic import ConfigDict, Field, model_validator

from langchain.chains.combine_documents.base import (
    BaseCombineDocumentsChain,
)
from langchain.chains.llm import LLMChain


def _get_default_document_prompt() -> PromptTemplate:
    return PromptTemplate(input_variables=["page_content"], template="{page_content}")


@deprecated(
    since="0.3.1",
    removal="1.0",
    message=(
        "This class is deprecated. Please see the migration guide here for "
        "a recommended replacement: "
        "https://python.langchain.com/docs/versions/migrating_chains/refine_docs_chain/"  # noqa: E501
    ),
)
class RefineDocumentsChain(BaseCombineDocumentsChain):
    """Combine documents by doing a first pass and then refining on more documents.

    This algorithm first calls `initial_llm_chain` on the first document, passing
    that first document in with the variable name `document_variable_name`, and
    produces a new variable with the variable name `initial_response_name`.

    Then, it loops over every remaining document. This is called the "refine" step.
    It calls `refine_llm_chain`,
    passing in that document with the variable name `document_variable_name`
    as well as the previous response with the variable name `initial_response_name`.

    Example:
        .. code-block:: python

            from langchain.chains import RefineDocumentsChain, LLMChain
            from langchain_core.prompts import PromptTemplate
            from langchain_community.llms import OpenAI

            # This controls how each document will be formatted. Specifically,
            # it will be passed to `format_document` - see that function for more
            # details.
            document_prompt = PromptTemplate(
                input_variables=["page_content"],
                 template="{page_content}"
            )
            document_variable_name = "context"
            llm = OpenAI()
            # The prompt here should take as an input variable the
            # `document_variable_name`
            prompt = PromptTemplate.from_template(
                "Summarize this content: {context}"
            )
            initial_llm_chain = LLMChain(llm=llm, prompt=prompt)
            initial_response_name = "prev_response"
            # The prompt here should take as an input variable the
            # `document_variable_name` as well as `initial_response_name`
            prompt_refine = PromptTemplate.from_template(
                "Here's your first summary: {prev_response}. "
                "Now add to it based on the following context: {context}"
            )
            refine_llm_chain = LLMChain(llm=llm, prompt=prompt_refine)
            chain = RefineDocumentsChain(
                initial_llm_chain=initial_llm_chain,
                refine_llm_chain=refine_llm_chain,
                document_prompt=document_prompt,
                document_variable_name=document_variable_name,
                initial_response_name=initial_response_name,
            )
    """

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
    """Prompt to use to format each document, gets passed to `format_document`."""
    return_intermediate_steps: bool = False
    """Return the results of the refine steps in the output."""

    @property
    def output_keys(self) -> list[str]:
        """Expect input key.

        :meta private:
        """
        _output_keys = super().output_keys
        if self.return_intermediate_steps:
            _output_keys = _output_keys + ["intermediate_steps"]
        return _output_keys

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    @model_validator(mode="before")
    @classmethod
    def get_return_intermediate_steps(cls, values: dict) -> Any:
        """For backwards compatibility."""
        if "return_refine_steps" in values:
            values["return_intermediate_steps"] = values["return_refine_steps"]
            del values["return_refine_steps"]
        return values

    @model_validator(mode="before")
    @classmethod
    def get_default_document_variable_name(cls, values: dict) -> Any:
        """Get default document variable name, if not provided."""
        if "initial_llm_chain" not in values:
            raise ValueError("initial_llm_chain must be provided")

        llm_chain_variables = values["initial_llm_chain"].prompt.input_variables
        if "document_variable_name" not in values:
            if len(llm_chain_variables) == 1:
                values["document_variable_name"] = llm_chain_variables[0]
            else:
                raise ValueError(
                    "document_variable_name must be provided if there are "
                    "multiple llm_chain input_variables"
                )
        else:
            if values["document_variable_name"] not in llm_chain_variables:
                raise ValueError(
                    f"document_variable_name {values['document_variable_name']} was "
                    f"not found in llm_chain input_variables: {llm_chain_variables}"
                )
        return values

    def combine_docs(
        self, docs: list[Document], callbacks: Callbacks = None, **kwargs: Any
    ) -> tuple[str, dict]:
        """Combine by mapping first chain over all, then stuffing into final chain.

        Args:
            docs: List of documents to combine
            callbacks: Callbacks to be passed through
            **kwargs: additional parameters to be passed to LLM calls (like other
                input variables besides the documents)

        Returns:
            The first element returned is the single string output. The second
            element returned is a dictionary of other keys to return.
        """
        inputs = self._construct_initial_inputs(docs, **kwargs)
        res = self.initial_llm_chain.predict(callbacks=callbacks, **inputs)
        refine_steps = [res]
        for doc in docs[1:]:
            base_inputs = self._construct_refine_inputs(doc, res)
            inputs = {**base_inputs, **kwargs}
            res = self.refine_llm_chain.predict(callbacks=callbacks, **inputs)
            refine_steps.append(res)
        return self._construct_result(refine_steps, res)

    async def acombine_docs(
        self, docs: list[Document], callbacks: Callbacks = None, **kwargs: Any
    ) -> tuple[str, dict]:
        """Async combine by mapping a first chain over all, then stuffing
         into a final chain.

        Args:
            docs: List of documents to combine
            callbacks: Callbacks to be passed through
            **kwargs: additional parameters to be passed to LLM calls (like other
                input variables besides the documents)

        Returns:
            The first element returned is the single string output. The second
            element returned is a dictionary of other keys to return.
        """
        inputs = self._construct_initial_inputs(docs, **kwargs)
        res = await self.initial_llm_chain.apredict(callbacks=callbacks, **inputs)
        refine_steps = [res]
        for doc in docs[1:]:
            base_inputs = self._construct_refine_inputs(doc, res)
            inputs = {**base_inputs, **kwargs}
            res = await self.refine_llm_chain.apredict(callbacks=callbacks, **inputs)
            refine_steps.append(res)
        return self._construct_result(refine_steps, res)

    def _construct_result(self, refine_steps: list[str], res: str) -> tuple[str, dict]:
        if self.return_intermediate_steps:
            extra_return_dict = {"intermediate_steps": refine_steps}
        else:
            extra_return_dict = {}
        return res, extra_return_dict

    def _construct_refine_inputs(self, doc: Document, res: str) -> dict[str, Any]:
        return {
            self.document_variable_name: format_document(doc, self.document_prompt),
            self.initial_response_name: res,
        }

    def _construct_initial_inputs(
        self, docs: list[Document], **kwargs: Any
    ) -> dict[str, Any]:
        base_info = {"page_content": docs[0].page_content}
        base_info.update(docs[0].metadata)
        document_info = {k: base_info[k] for k in self.document_prompt.input_variables}
        base_inputs: dict = {
            self.document_variable_name: self.document_prompt.format(**document_info)
        }
        inputs = {**base_inputs, **kwargs}
        return inputs

    @property
    def _chain_type(self) -> str:
        return "refine_documents_chain"
