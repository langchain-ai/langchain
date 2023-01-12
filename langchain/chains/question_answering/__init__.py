"""Load question answering chains."""
from typing import Any, Mapping, Optional, Protocol

from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain
from langchain.chains.combine_documents.map_rerank import MapRerankDocumentsChain
from langchain.chains.combine_documents.refine import RefineDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import (
    map_reduce_prompt,
    map_rerank_prompt,
    refine_prompts,
    stuff_prompt,
)
from langchain.llms.base import BaseLLM
from langchain.prompts.base import BasePromptTemplate


class LoadingCallable(Protocol):
    """Interface for loading the combine documents chain."""

    def __call__(self, llm: BaseLLM, **kwargs: Any) -> BaseCombineDocumentsChain:
        """Callable to load the combine documents chain."""


def _load_map_rerank_chain(
    llm: BaseLLM,
    prompt: BasePromptTemplate = map_rerank_prompt.PROMPT,
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
    llm: BaseLLM,
    prompt: BasePromptTemplate = stuff_prompt.PROMPT,
    document_variable_name: str = "context",
    verbose: Optional[bool] = None,
    **kwargs: Any,
) -> StuffDocumentsChain:
    llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=verbose)
    # TODO: document prompt
    return StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name=document_variable_name,
        verbose=verbose,
        **kwargs,
    )


def _load_map_reduce_chain(
    llm: BaseLLM,
    question_prompt: BasePromptTemplate = map_reduce_prompt.QUESTION_PROMPT,
    combine_prompt: BasePromptTemplate = map_reduce_prompt.COMBINE_PROMPT,
    combine_document_variable_name: str = "summaries",
    map_reduce_document_variable_name: str = "context",
    collapse_prompt: Optional[BasePromptTemplate] = None,
    reduce_llm: Optional[BaseLLM] = None,
    collapse_llm: Optional[BaseLLM] = None,
    verbose: Optional[bool] = None,
    **kwargs: Any,
) -> MapReduceDocumentsChain:
    map_chain = LLMChain(llm=llm, prompt=question_prompt, verbose=verbose)
    _reduce_llm = reduce_llm or llm
    reduce_chain = LLMChain(llm=_reduce_llm, prompt=combine_prompt, verbose=verbose)
    # TODO: document prompt
    combine_document_chain = StuffDocumentsChain(
        llm_chain=reduce_chain,
        document_variable_name=combine_document_variable_name,
        verbose=verbose,
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
                verbose=verbose,
            ),
            document_variable_name=combine_document_variable_name,
        )
    return MapReduceDocumentsChain(
        llm_chain=map_chain,
        combine_document_chain=combine_document_chain,
        document_variable_name=map_reduce_document_variable_name,
        collapse_document_chain=collapse_chain,
        verbose=verbose,
        **kwargs,
    )


def _load_refine_chain(
    llm: BaseLLM,
    question_prompt: BasePromptTemplate = refine_prompts.DEFAULT_TEXT_QA_PROMPT,
    refine_prompt: BasePromptTemplate = refine_prompts.DEFAULT_REFINE_PROMPT,
    document_variable_name: str = "context_str",
    initial_response_name: str = "existing_answer",
    refine_llm: Optional[BaseLLM] = None,
    verbose: Optional[bool] = None,
    **kwargs: Any,
) -> RefineDocumentsChain:
    initial_chain = LLMChain(llm=llm, prompt=question_prompt, verbose=verbose)
    _refine_llm = refine_llm or llm
    refine_chain = LLMChain(llm=_refine_llm, prompt=refine_prompt, verbose=verbose)
    return RefineDocumentsChain(
        initial_llm_chain=initial_chain,
        refine_llm_chain=refine_chain,
        document_variable_name=document_variable_name,
        initial_response_name=initial_response_name,
        verbose=verbose,
        **kwargs,
    )


def load_qa_chain(
    llm: BaseLLM,
    chain_type: str = "stuff",
    verbose: Optional[bool] = None,
    **kwargs: Any,
) -> BaseCombineDocumentsChain:
    """Load question answering chain.

    Args:
        llm: Language Model to use in the chain.
        chain_type: Type of document combining chain to use. Should be one of "stuff",
            "map_reduce", and "refine".
        verbose: Whether chains should be run in verbose mode or not. Note that this
            applies to all chains that make up the final chain.

    Returns:
        A chain to use for question answering.
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
    return loader_mapping[chain_type](llm, verbose=verbose, **kwargs)
