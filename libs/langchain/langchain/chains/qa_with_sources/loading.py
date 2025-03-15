"""Load question answering with sources chains."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Protocol

from langchain_core._api import deprecated
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate

from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain
from langchain.chains.combine_documents.map_rerank import MapRerankDocumentsChain
from langchain.chains.combine_documents.reduce import ReduceDocumentsChain
from langchain.chains.combine_documents.refine import RefineDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chains.qa_with_sources import (
    map_reduce_prompt,
    refine_prompts,
    stuff_prompt,
)
from langchain.chains.question_answering.map_rerank_prompt import (
    PROMPT as MAP_RERANK_PROMPT,
)


class LoadingCallable(Protocol):
    """Interface for loading the combine documents chain."""

    def __call__(
        self, llm: BaseLanguageModel, **kwargs: Any
    ) -> BaseCombineDocumentsChain:
        """Callable to load the combine documents chain."""


def _load_map_rerank_chain(
    llm: BaseLanguageModel,
    prompt: BasePromptTemplate = MAP_RERANK_PROMPT,
    verbose: bool = False,
    document_variable_name: str = "context",
    rank_key: str = "score",
    answer_key: str = "answer",
    **kwargs: Any,
) -> MapRerankDocumentsChain:
    llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=verbose)
    return MapRerankDocumentsChain(
        llm_chain=llm_chain,
        rank_key=rank_key,
        answer_key=answer_key,
        document_variable_name=document_variable_name,
        **kwargs,
    )


def _load_stuff_chain(
    llm: BaseLanguageModel,
    prompt: BasePromptTemplate = stuff_prompt.PROMPT,
    document_prompt: BasePromptTemplate = stuff_prompt.EXAMPLE_PROMPT,
    document_variable_name: str = "summaries",
    verbose: Optional[bool] = None,
    **kwargs: Any,
) -> StuffDocumentsChain:
    llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=verbose)  # type: ignore[arg-type]
    return StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name=document_variable_name,
        document_prompt=document_prompt,
        verbose=verbose,  # type: ignore[arg-type]
        **kwargs,
    )


def _load_map_reduce_chain(
    llm: BaseLanguageModel,
    question_prompt: BasePromptTemplate = map_reduce_prompt.QUESTION_PROMPT,
    combine_prompt: BasePromptTemplate = map_reduce_prompt.COMBINE_PROMPT,
    document_prompt: BasePromptTemplate = map_reduce_prompt.EXAMPLE_PROMPT,
    combine_document_variable_name: str = "summaries",
    map_reduce_document_variable_name: str = "context",
    collapse_prompt: Optional[BasePromptTemplate] = None,
    reduce_llm: Optional[BaseLanguageModel] = None,
    collapse_llm: Optional[BaseLanguageModel] = None,
    verbose: Optional[bool] = None,
    token_max: int = 3000,
    **kwargs: Any,
) -> MapReduceDocumentsChain:
    map_chain = LLMChain(llm=llm, prompt=question_prompt, verbose=verbose)  # type: ignore[arg-type]
    _reduce_llm = reduce_llm or llm
    reduce_chain = LLMChain(llm=_reduce_llm, prompt=combine_prompt, verbose=verbose)  # type: ignore[arg-type]
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain,
        document_variable_name=combine_document_variable_name,
        document_prompt=document_prompt,
        verbose=verbose,  # type: ignore[arg-type]
    )
    if collapse_prompt is None:
        collapse_chain = None
        if collapse_llm is not None:
            raise ValueError(
                "collapse_llm provided, but collapse_prompt was not: please "
                "provide one or stop providing collapse_llm."
            )
    else:
        _collapse_llm = collapse_llm or llm
        collapse_chain = StuffDocumentsChain(
            llm_chain=LLMChain(
                llm=_collapse_llm,
                prompt=collapse_prompt,
                verbose=verbose,  # type: ignore[arg-type]
            ),
            document_variable_name=combine_document_variable_name,
            document_prompt=document_prompt,
        )
    reduce_documents_chain = ReduceDocumentsChain(
        combine_documents_chain=combine_documents_chain,
        collapse_documents_chain=collapse_chain,
        token_max=token_max,
        verbose=verbose,  # type: ignore[arg-type]
    )
    return MapReduceDocumentsChain(
        llm_chain=map_chain,
        reduce_documents_chain=reduce_documents_chain,
        document_variable_name=map_reduce_document_variable_name,
        verbose=verbose,  # type: ignore[arg-type]
        **kwargs,
    )


def _load_refine_chain(
    llm: BaseLanguageModel,
    question_prompt: BasePromptTemplate = refine_prompts.DEFAULT_TEXT_QA_PROMPT,
    refine_prompt: BasePromptTemplate = refine_prompts.DEFAULT_REFINE_PROMPT,
    document_prompt: BasePromptTemplate = refine_prompts.EXAMPLE_PROMPT,
    document_variable_name: str = "context_str",
    initial_response_name: str = "existing_answer",
    refine_llm: Optional[BaseLanguageModel] = None,
    verbose: Optional[bool] = None,
    **kwargs: Any,
) -> RefineDocumentsChain:
    initial_chain = LLMChain(llm=llm, prompt=question_prompt, verbose=verbose)  # type: ignore[arg-type]
    _refine_llm = refine_llm or llm
    refine_chain = LLMChain(llm=_refine_llm, prompt=refine_prompt, verbose=verbose)  # type: ignore[arg-type]
    return RefineDocumentsChain(
        initial_llm_chain=initial_chain,
        refine_llm_chain=refine_chain,
        document_variable_name=document_variable_name,
        initial_response_name=initial_response_name,
        document_prompt=document_prompt,
        verbose=verbose,  # type: ignore[arg-type]
        **kwargs,
    )


@deprecated(
    since="0.2.13",
    removal="1.0",
    message=(
        "This function is deprecated. Refer to this guide on retrieval and question "
        "answering with sources: "
        "https://python.langchain.com/docs/how_to/qa_sources/"
        "\nSee also the following migration guides for replacements "
        "based on `chain_type`:\n"
        "stuff: https://python.langchain.com/docs/versions/migrating_chains/stuff_docs_chain\n"  # noqa: E501
        "map_reduce: https://python.langchain.com/docs/versions/migrating_chains/map_reduce_chain\n"  # noqa: E501
        "refine: https://python.langchain.com/docs/versions/migrating_chains/refine_chain\n"  # noqa: E501
        "map_rerank: https://python.langchain.com/docs/versions/migrating_chains/map_rerank_docs_chain\n"  # noqa: E501
    ),
)
def load_qa_with_sources_chain(
    llm: BaseLanguageModel,
    chain_type: str = "stuff",
    verbose: Optional[bool] = None,
    **kwargs: Any,
) -> BaseCombineDocumentsChain:
    """Load a question answering with sources chain.

    Args:
        llm: Language Model to use in the chain.
        chain_type: Type of document combining chain to use. Should be one of "stuff",
            "map_reduce", "refine" and "map_rerank".
        verbose: Whether chains should be run in verbose mode or not. Note that this
            applies to all chains that make up the final chain.

    Returns:
        A chain to use for question answering with sources.
    """
    loader_mapping: Mapping[str, LoadingCallable] = {
        "stuff": _load_stuff_chain,
        "map_reduce": _load_map_reduce_chain,
        "refine": _load_refine_chain,
        "map_rerank": _load_map_rerank_chain,
    }
    if chain_type not in loader_mapping:
        raise ValueError(
            f"Got unsupported chain type: {chain_type}. "
            f"Should be one of {loader_mapping.keys()}"
        )
    _func: LoadingCallable = loader_mapping[chain_type]
    return _func(llm, verbose=verbose, **kwargs)
