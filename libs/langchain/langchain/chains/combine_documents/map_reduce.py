"""Combining documents by mapping a chain over them first, then combining results."""

from __future__ import annotations

from operator import itemgetter
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

from langchain_core.documents import Document
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import BasePromptTemplate, PromptTemplate, format_document
from langchain_core.pydantic_v1 import BaseModel, Extra, create_model, root_validator
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.runnables.config import RunnableConfig

from langchain.callbacks.manager import Callbacks
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.chains.combine_documents.reduce import ReduceDocumentsChain
from langchain.chains.llm import LLMChain

LanguageModelLike = Union[
    Runnable[LanguageModelInput, str], Runnable[LanguageModelInput, BaseMessage]
]


def create_map_documents_chain(
    llm: LanguageModelLike,
    prompt: BasePromptTemplate,
    *,
    document_input_key: str,
    document_prompt: Optional[BasePromptTemplate] = None,
) -> Runnable[Dict[str, Any], List[Document]]:
    _document_prompt = document_prompt or PromptTemplate.from_template("{page_content}")

    def _format_document(inputs: dict) -> str:
        return format_document(inputs[document_input_key], _document_prompt)

    map_content_chain = (
        RunnablePassthrough.assign(**{document_input_key: _format_document})
        | prompt
        | llm
        | StrOutputParser()
    )

    map_doc_chain = RunnableParallel(
        doc=itemgetter(document_input_key), content=map_content_chain
    ) | (lambda x: Document(page_content=x["content"], metadata=x["doc"].metadata))

    def list_inputs(inputs: dict) -> list:
        docs = inputs.pop(document_input_key)
        inputs = {k: v for k, v in inputs.items() for k in prompt.input_variables}
        return [{document_input_key: doc, **inputs} for doc in docs]

    return list_inputs | map_doc_chain.map()


def create_reduce_documents_chain(
    llm: LanguageModelLike,
    prompt: BasePromptTemplate,
    *,
    document_input_key: str,
    document_prompt: Optional[BasePromptTemplate] = None,
    document_separator: str = "\n\n",
) -> Runnable:
    _document_prompt = document_prompt or PromptTemplate.from_template("{page_content}")

    def _format_inputs(inputs: dict) -> dict:
        docs = inputs[document_input_key]
        inputs[document_input_key] = document_separator.join(
            format_document(doc, _document_prompt) for doc in docs
        )
        return {k: v for k, v in inputs.items() if k in prompt.input_variables}

    return _format_inputs | prompt | llm | StrOutputParser()


def create_collapse_documents_chain(
    llm: LanguageModelLike,
    prompt: BasePromptTemplate,
    *,
    document_input_key: str,
    document_prompt: Optional[BasePromptTemplate] = None,
    document_separator: str = "\n\n",
    token_max: int = 4000,
    token_len_func: Optional[Callable[[str], int]] = None,
) -> Runnable:
    _document_prompt = document_prompt or PromptTemplate.from_template("{page_content}")
    _token_len_func = token_len_func or getattr(llm, "get_num_tokens", len)

    def _format_inputs(inputs: dict) -> dict:
        docs = inputs[document_input_key]
        inputs[document_input_key] = document_separator.join(
            format_document(doc, _document_prompt) for doc in docs
        )
        return {k: v for k, v in inputs.items() if k in prompt.input_variables}

    def _format_metadata(inputs: dict) -> dict:
        docs = inputs[document_input_key]
        combined_metadata = {k: str(v) for k, v in docs[0].metadata.items()}
        for doc in docs[1:]:
            for k, v in doc.metadata.items():
                if k in combined_metadata:
                    combined_metadata[k] += f", {v}"
                else:
                    combined_metadata[k] = str(v)
        return combined_metadata

    reduce_chain = RunnableParallel(
        page_content=_format_inputs | prompt | llm | StrOutputParser(),
        metadata=_format_metadata,
    ) | (lambda x: Document(page_content=x["page_content"], metadata=x["metadata"]))

    def _get_num_tokens(docs):
        formatted = document_separator.join(
            format_document(doc, _document_prompt) for doc in docs
        )
        return _token_len_func(formatted)

    def _partition_docs(inputs):
        docs = inputs.pop(document_input_key)
        partitions = []
        curr = []
        curr_len = 0
        for doc in docs:
            # Add empty doc so document separator is included in formatted string.
            doc_len = _get_num_tokens([Document(page_content=""), doc])
            if doc_len > token_max:
                raise ValueError
            elif curr_len + doc_len > token_max:
                partitions.append(curr)
                curr = []
                curr_len = 0
            else:
                curr.append(doc)
                curr_len += doc_len
        if curr:
            partitions.append(curr)
        return [{document_input_key: docs, **inputs} for docs in partitions]

    def collapse(inputs: dict) -> Union[List[Document], Runnable]:
        docs = inputs[document_input_key]
        while _get_num_tokens(docs) > token_max:
            return (
                RunnableParallel(
                    **{document_input_key: _partition_docs | reduce_chain.map()}
                )
                | collapse
            )
        else:
            return docs

    return RunnableLambda(collapse)


def create_map_reduce_documents_chain(
    map_chain: Runnable[Dict[str, Any], List[Document]],
    reduce_chain: Runnable[Dict[str, Any], str],
    *,
    document_input_key: str,
    collapse_chain: Optional[Runnable[Dict[str, Any], List[Document]]] = None,
) -> Runnable:
    if not collapse_chain:
        return (
            RunnablePassthrough.assign(**{document_input_key: map_chain}) | reduce_chain
        )
    else:
        return (
            RunnablePassthrough.assign(**{document_input_key: map_chain})
            | RunnablePassthrough.assign(**{document_input_key: collapse_chain})
            | reduce_chain
        )


class MapReduceDocumentsChain(BaseCombineDocumentsChain):
    """Combining documents by mapping a chain over them, then combining results.

    We first call `llm_chain` on each document individually, passing in the
    `page_content` and any other kwargs. This is the `map` step.

    We then process the results of that `map` step in a `reduce` step. This should
    likely be a ReduceDocumentsChain.

    Example:
        .. code-block:: python

            from langchain.chains import (
                StuffDocumentsChain,
                LLMChain,
                ReduceDocumentsChain,
                MapReduceDocumentsChain,
            )
            from langchain_core.prompts import PromptTemplate
            from langchain.llms import OpenAI

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
            llm_chain = LLMChain(llm=llm, prompt=prompt)
            # We now define how to combine these summaries
            reduce_prompt = PromptTemplate.from_template(
                "Combine these summaries: {context}"
            )
            reduce_llm_chain = LLMChain(llm=llm, prompt=reduce_prompt)
            combine_documents_chain = StuffDocumentsChain(
                llm_chain=reduce_llm_chain,
                document_prompt=document_prompt,
                document_variable_name=document_variable_name
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
            prompt = PromptTemplate.from_template(
                "Collapse this content: {context}"
            )
            llm_chain = LLMChain(llm=llm, prompt=prompt)
            collapse_documents_chain = StuffDocumentsChain(
                llm_chain=llm_chain,
                document_prompt=document_prompt,
                document_variable_name=document_variable_name
            )
            reduce_documents_chain = ReduceDocumentsChain(
                combine_documents_chain=combine_documents_chain,
                collapse_documents_chain=collapse_documents_chain,
            )
            chain = MapReduceDocumentsChain(
                llm_chain=llm_chain,
                reduce_documents_chain=reduce_documents_chain,
            )
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

    def get_output_schema(
        self, config: Optional[RunnableConfig] = None
    ) -> Type[BaseModel]:
        if self.return_intermediate_steps:
            return create_model(
                "MapReduceDocumentsOutput",
                **{
                    self.output_key: (str, None),
                    "intermediate_steps": (List[str], None),
                },  # type: ignore[call-overload]
            )

        return super().get_output_schema(config)

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
            if "reduce_documents_chain" in values:
                raise ValueError(
                    "Both `reduce_documents_chain` and `combine_document_chain` "
                    "cannot be provided at the same time. `combine_document_chain` "
                    "is deprecated, please only provide `reduce_documents_chain`"
                )
            combine_chain = values["combine_document_chain"]
            collapse_chain = values.get("collapse_document_chain")
            reduce_chain = ReduceDocumentsChain(
                combine_documents_chain=combine_chain,
                collapse_documents_chain=collapse_chain,
            )
            values["reduce_documents_chain"] = reduce_chain
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
        """Kept for backward compatibility."""
        if isinstance(self.reduce_documents_chain, ReduceDocumentsChain):
            if self.reduce_documents_chain.collapse_documents_chain:
                return self.reduce_documents_chain.collapse_documents_chain
            else:
                return self.reduce_documents_chain.combine_documents_chain
        else:
            raise ValueError(
                f"`reduce_documents_chain` is of type "
                f"{type(self.reduce_documents_chain)} so it does not have "
                f"this attribute."
            )

    @property
    def combine_document_chain(self) -> BaseCombineDocumentsChain:
        """Kept for backward compatibility."""
        if isinstance(self.reduce_documents_chain, ReduceDocumentsChain):
            return self.reduce_documents_chain.combine_documents_chain
        else:
            raise ValueError(
                f"`reduce_documents_chain` is of type "
                f"{type(self.reduce_documents_chain)} so it does not have "
                f"this attribute."
            )

    def combine_docs(
        self,
        docs: List[Document],
        token_max: Optional[int] = None,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Tuple[str, dict]:
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
            result_docs, token_max=token_max, callbacks=callbacks, **kwargs
        )
        if self.return_intermediate_steps:
            intermediate_steps = [r[question_result_key] for r in map_results]
            extra_return_dict["intermediate_steps"] = intermediate_steps
        return result, extra_return_dict

    async def acombine_docs(
        self,
        docs: List[Document],
        token_max: Optional[int] = None,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Tuple[str, dict]:
        """Combine documents in a map reduce manner.

        Combine by mapping first chain over all documents, then reducing the results.
        This reducing can be done recursively if needed (if there are many documents).
        """
        map_results = await self.llm_chain.aapply(
            # FYI - this is parallelized and so it is fast.
            [{**{self.document_variable_name: d.page_content}, **kwargs} for d in docs],
            callbacks=callbacks,
        )
        question_result_key = self.llm_chain.output_key
        result_docs = [
            Document(page_content=r[question_result_key], metadata=docs[i].metadata)
            # This uses metadata from the docs, and the textual results from `results`
            for i, r in enumerate(map_results)
        ]
        result, extra_return_dict = await self.reduce_documents_chain.acombine_docs(
            result_docs, token_max=token_max, callbacks=callbacks, **kwargs
        )
        if self.return_intermediate_steps:
            intermediate_steps = [r[question_result_key] for r in map_results]
            extra_return_dict["intermediate_steps"] = intermediate_steps
        return result, extra_return_dict

    @property
    def _chain_type(self) -> str:
        return "map_reduce_documents_chain"
