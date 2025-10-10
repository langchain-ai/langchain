"""Combining documents by mapping a chain over them first, then combining results."""

from __future__ import annotations

from typing import Any

from langchain_core._api import deprecated
from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document
from langchain_core.runnables.config import RunnableConfig
from langchain_core.utils.pydantic import create_model
from pydantic import BaseModel, ConfigDict, model_validator
from typing_extensions import override

from langchain_classic.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain_classic.chains.combine_documents.reduce import ReduceDocumentsChain
from langchain_classic.chains.llm import LLMChain


@deprecated(
    since="0.3.1",
    removal="1.0",
    message=(
        "This class is deprecated. Please see the migration guide here for "
        "a recommended replacement: "
        "https://python.langchain.com/docs/versions/migrating_chains/map_reduce_chain/"
    ),
)
class MapReduceDocumentsChain(BaseCombineDocumentsChain):
    """Combining documents by mapping a chain over them, then combining results.

    We first call `llm_chain` on each document individually, passing in the
    `page_content` and any other kwargs. This is the `map` step.

    We then process the results of that `map` step in a `reduce` step. This should
    likely be a ReduceDocumentsChain.

    Example:
        ```python
        from langchain_classic.chains import (
            StuffDocumentsChain,
            LLMChain,
            ReduceDocumentsChain,
            MapReduceDocumentsChain,
        )
        from langchain_core.prompts import PromptTemplate
        from langchain_community.llms import OpenAI

        # This controls how each document will be formatted. Specifically,
        # it will be passed to `format_document` - see that function for more
        # details.
        document_prompt = PromptTemplate(
            input_variables=["page_content"], template="{page_content}"
        )
        document_variable_name = "context"
        model = OpenAI()
        # The prompt here should take as an input variable the
        # `document_variable_name`
        prompt = PromptTemplate.from_template("Summarize this content: {context}")
        llm_chain = LLMChain(llm=model, prompt=prompt)
        # We now define how to combine these summaries
        reduce_prompt = PromptTemplate.from_template(
            "Combine these summaries: {context}"
        )
        reduce_llm_chain = LLMChain(llm=model, prompt=reduce_prompt)
        combine_documents_chain = StuffDocumentsChain(
            llm_chain=reduce_llm_chain,
            document_prompt=document_prompt,
            document_variable_name=document_variable_name,
        )
        reduce_documents_chain = ReduceDocumentsChain(
            combine_documents_chain=combine_documents_chain,
        )
        chain = MapReduceDocumentsChain(
            llm_chain=llm_chain,
            reduce_documents_chain=reduce_documents_chain,
        )
        # If we wanted to, we could also pass in collapse_documents_chain
        # which is specifically aimed at collapsing documents BEFORE
        # the final call.
        prompt = PromptTemplate.from_template("Collapse this content: {context}")
        llm_chain = LLMChain(llm=model, prompt=prompt)
        collapse_documents_chain = StuffDocumentsChain(
            llm_chain=llm_chain,
            document_prompt=document_prompt,
            document_variable_name=document_variable_name,
        )
        reduce_documents_chain = ReduceDocumentsChain(
            combine_documents_chain=combine_documents_chain,
            collapse_documents_chain=collapse_documents_chain,
        )
        chain = MapReduceDocumentsChain(
            llm_chain=llm_chain,
            reduce_documents_chain=reduce_documents_chain,
        )
        ```
    """

    llm_chain: LLMChain
    """Chain to apply to each document individually."""
    reduce_documents_chain: BaseCombineDocumentsChain
    """Chain to use to reduce the results of applying `llm_chain` to each doc.
    This typically either a ReduceDocumentChain or StuffDocumentChain."""
    document_variable_name: str
    """The variable name in the llm_chain to put the documents in.
    If only one variable in the llm_chain, this need not be provided."""
    return_intermediate_steps: bool = False
    """Return the results of the map steps in the output."""

    @override
    def get_output_schema(
        self,
        config: RunnableConfig | None = None,
    ) -> type[BaseModel]:
        if self.return_intermediate_steps:
            return create_model(
                "MapReduceDocumentsOutput",
                **{
                    self.output_key: (str, None),
                    "intermediate_steps": (list[str], None),
                },
            )

        return super().get_output_schema(config)

    @property
    def output_keys(self) -> list[str]:
        """Expect input key.

        :meta private:
        """
        _output_keys = super().output_keys
        if self.return_intermediate_steps:
            _output_keys = [*_output_keys, "intermediate_steps"]
        return _output_keys

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    @model_validator(mode="before")
    @classmethod
    def get_reduce_chain(cls, values: dict) -> Any:
        """For backwards compatibility."""
        if "combine_document_chain" in values:
            if "reduce_documents_chain" in values:
                msg = (
                    "Both `reduce_documents_chain` and `combine_document_chain` "
                    "cannot be provided at the same time. `combine_document_chain` "
                    "is deprecated, please only provide `reduce_documents_chain`"
                )
                raise ValueError(msg)
            combine_chain = values["combine_document_chain"]
            collapse_chain = values.get("collapse_document_chain")
            reduce_chain = ReduceDocumentsChain(
                combine_documents_chain=combine_chain,
                collapse_documents_chain=collapse_chain,
            )
            values["reduce_documents_chain"] = reduce_chain
            del values["combine_document_chain"]
            values.pop("collapse_document_chain", None)

        return values

    @model_validator(mode="before")
    @classmethod
    def get_return_intermediate_steps(cls, values: dict) -> Any:
        """For backwards compatibility."""
        if "return_map_steps" in values:
            values["return_intermediate_steps"] = values["return_map_steps"]
            del values["return_map_steps"]
        return values

    @model_validator(mode="before")
    @classmethod
    def get_default_document_variable_name(cls, values: dict) -> Any:
        """Get default document variable name, if not provided."""
        if "llm_chain" not in values:
            msg = "llm_chain must be provided"
            raise ValueError(msg)

        llm_chain_variables = values["llm_chain"].prompt.input_variables
        if "document_variable_name" not in values:
            if len(llm_chain_variables) == 1:
                values["document_variable_name"] = llm_chain_variables[0]
            else:
                msg = (
                    "document_variable_name must be provided if there are "
                    "multiple llm_chain input_variables"
                )
                raise ValueError(msg)
        elif values["document_variable_name"] not in llm_chain_variables:
            msg = (
                f"document_variable_name {values['document_variable_name']} was "
                f"not found in llm_chain input_variables: {llm_chain_variables}"
            )
            raise ValueError(msg)
        return values

    @property
    def collapse_document_chain(self) -> BaseCombineDocumentsChain:
        """Kept for backward compatibility."""
        if isinstance(self.reduce_documents_chain, ReduceDocumentsChain):
            if self.reduce_documents_chain.collapse_documents_chain:
                return self.reduce_documents_chain.collapse_documents_chain
            return self.reduce_documents_chain.combine_documents_chain
        msg = (
            f"`reduce_documents_chain` is of type "
            f"{type(self.reduce_documents_chain)} so it does not have "
            f"this attribute."
        )
        raise ValueError(msg)

    @property
    def combine_document_chain(self) -> BaseCombineDocumentsChain:
        """Kept for backward compatibility."""
        if isinstance(self.reduce_documents_chain, ReduceDocumentsChain):
            return self.reduce_documents_chain.combine_documents_chain
        msg = (
            f"`reduce_documents_chain` is of type "
            f"{type(self.reduce_documents_chain)} so it does not have "
            f"this attribute."
        )
        raise ValueError(msg)

    def combine_docs(
        self,
        docs: list[Document],
        token_max: int | None = None,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> tuple[str, dict]:
        """Combine documents in a map reduce manner.

        Combine by mapping first chain over all documents, then reducing the results.
        This reducing can be done recursively if needed (if there are many documents).
        """
        map_results = self.llm_chain.apply(
            # FYI - this is parallelized and so it is fast.
            [{self.document_variable_name: d.page_content, **kwargs} for d in docs],
            callbacks=callbacks,
        )
        question_result_key = self.llm_chain.output_key
        result_docs = [
            Document(page_content=r[question_result_key], metadata=docs[i].metadata)
            # This uses metadata from the docs, and the textual results from `results`
            for i, r in enumerate(map_results)
        ]
        result, extra_return_dict = self.reduce_documents_chain.combine_docs(
            result_docs,
            token_max=token_max,
            callbacks=callbacks,
            **kwargs,
        )
        if self.return_intermediate_steps:
            intermediate_steps = [r[question_result_key] for r in map_results]
            extra_return_dict["intermediate_steps"] = intermediate_steps
        return result, extra_return_dict

    async def acombine_docs(
        self,
        docs: list[Document],
        token_max: int | None = None,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> tuple[str, dict]:
        """Combine documents in a map reduce manner.

        Combine by mapping first chain over all documents, then reducing the results.
        This reducing can be done recursively if needed (if there are many documents).
        """
        map_results = await self.llm_chain.aapply(
            # FYI - this is parallelized and so it is fast.
            [{self.document_variable_name: d.page_content, **kwargs} for d in docs],
            callbacks=callbacks,
        )
        question_result_key = self.llm_chain.output_key
        result_docs = [
            Document(page_content=r[question_result_key], metadata=docs[i].metadata)
            # This uses metadata from the docs, and the textual results from `results`
            for i, r in enumerate(map_results)
        ]
        result, extra_return_dict = await self.reduce_documents_chain.acombine_docs(
            result_docs,
            token_max=token_max,
            callbacks=callbacks,
            **kwargs,
        )
        if self.return_intermediate_steps:
            intermediate_steps = [r[question_result_key] for r in map_results]
            extra_return_dict["intermediate_steps"] = intermediate_steps
        return result, extra_return_dict

    @property
    def _chain_type(self) -> str:
        return "map_reduce_documents_chain"
