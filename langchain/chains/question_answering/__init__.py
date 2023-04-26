"""Load question answering chains."""
from typing import Any, Mapping, Optional, Protocol

from langchain.callbacks.base import BaseCallbackManager
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
from langchain.prompts.base import BasePromptTemplate
from langchain.schema import BaseLanguageModel


class LoadingCallable(Protocol):
    """Interface for loading the combine documents chain."""

    def __call__(
        self, llm: BaseLanguageModel, **kwargs: Any
    ) -> BaseCombineDocumentsChain:
        """Callable to load the combine documents chain."""


def _load_map_rerank_chain(
    llm: BaseLanguageModel,
    prompt: BasePromptTemplate = map_rerank_prompt.PROMPT,
    verbose: bool = False,
    document_variable_name: str = "context",
    rank_key: str = "score",
    answer_key: str = "answer",
    callback_manager: Optional[BaseCallbackManager] = None,
    **kwargs: Any,
) -> MapRerankDocumentsChain:
    llm_chain = LLMChain(
        llm=llm, prompt=prompt, verbose=verbose, callback_manager=callback_manager
    )
    return MapRerankDocumentsChain(
        llm_chain=llm_chain,
        rank_key=rank_key,
        answer_key=answer_key,
        document_variable_name=document_variable_name,
        verbose=verbose,
        callback_manager=callback_manager,
        **kwargs,
    )


def _load_stuff_chain(
    llm: BaseLanguageModel,
    prompt: Optional[BasePromptTemplate] = None,
    document_variable_name: str = "context",
    verbose: Optional[bool] = None,
    callback_manager: Optional[BaseCallbackManager] = None,
    **kwargs: Any,
) -> StuffDocumentsChain:
    _prompt = prompt or stuff_prompt.PROMPT_SELECTOR.get_prompt(llm)
    llm_chain = LLMChain(
        llm=llm, prompt=_prompt, verbose=verbose, callback_manager=callback_manager
    )
    # TODO: document prompt
    return StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name=document_variable_name,
        verbose=verbose,
        callback_manager=callback_manager,
        **kwargs,
    )


def _load_map_reduce_chain(
    llm: BaseLanguageModel,
    question_prompt: Optional[BasePromptTemplate] = None,
    combine_prompt: Optional[BasePromptTemplate] = None,
    combine_document_variable_name: str = "summaries",
    map_reduce_document_variable_name: str = "context",
    collapse_prompt: Optional[BasePromptTemplate] = None,
    reduce_llm: Optional[BaseLanguageModel] = None,
    collapse_llm: Optional[BaseLanguageModel] = None,
    verbose: Optional[bool] = None,
    callback_manager: Optional[BaseCallbackManager] = None,
    **kwargs: Any,
) -> MapReduceDocumentsChain:
    _question_prompt = (
        question_prompt or map_reduce_prompt.QUESTION_PROMPT_SELECTOR.get_prompt(llm)
    )
    _combine_prompt = (
        combine_prompt or map_reduce_prompt.COMBINE_PROMPT_SELECTOR.get_prompt(llm)
    )
    map_chain = LLMChain(
        llm=llm,
        prompt=_question_prompt,
        verbose=verbose,
        callback_manager=callback_manager,
    )
    _reduce_llm = reduce_llm or llm
    reduce_chain = LLMChain(
        llm=_reduce_llm,
        prompt=_combine_prompt,
        verbose=verbose,
        callback_manager=callback_manager,
    )
    # TODO: document prompt
    combine_document_chain = StuffDocumentsChain(
        llm_chain=reduce_chain,
        document_variable_name=combine_document_variable_name,
        verbose=verbose,
        callback_manager=callback_manager,
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
                callback_manager=callback_manager,
            ),
            document_variable_name=combine_document_variable_name,
            verbose=verbose,
            callback_manager=callback_manager,
        )
    return MapReduceDocumentsChain(
        llm_chain=map_chain,
        combine_document_chain=combine_document_chain,
        document_variable_name=map_reduce_document_variable_name,
        collapse_document_chain=collapse_chain,
        verbose=verbose,
        callback_manager=callback_manager,
        **kwargs,
    )


def _load_refine_chain(
    llm: BaseLanguageModel,
    question_prompt: Optional[BasePromptTemplate] = None,
    refine_prompt: Optional[BasePromptTemplate] = None,
    document_variable_name: str = "context_str",
    initial_response_name: str = "existing_answer",
    refine_llm: Optional[BaseLanguageModel] = None,
    verbose: Optional[bool] = None,
    callback_manager: Optional[BaseCallbackManager] = None,
    **kwargs: Any,
) -> RefineDocumentsChain:
    _question_prompt = (
        question_prompt or refine_prompts.QUESTION_PROMPT_SELECTOR.get_prompt(llm)
    )
    _refine_prompt = refine_prompt or refine_prompts.REFINE_PROMPT_SELECTOR.get_prompt(
        llm
    )
    initial_chain = LLMChain(
        llm=llm,
        prompt=_question_prompt,
        verbose=verbose,
        callback_manager=callback_manager,
    )
    _refine_llm = refine_llm or llm
    refine_chain = LLMChain(
        llm=_refine_llm,
        prompt=_refine_prompt,
        verbose=verbose,
        callback_manager=callback_manager,
    )
    return RefineDocumentsChain(
        initial_llm_chain=initial_chain,
        refine_llm_chain=refine_chain,
        document_variable_name=document_variable_name,
        initial_response_name=initial_response_name,
        verbose=verbose,
        callback_manager=callback_manager,
        **kwargs,
    )


def load_qa_chain(
    llm: BaseLanguageModel,
    chain_type: str = "stuff",
    verbose: Optional[bool] = None,
    callback_manager: Optional[BaseCallbackManager] = None,
    **kwargs: Any,
) -> BaseCombineDocumentsChain:
    """Load question answering chain.

    Args:
        llm: Language Model to use in the chain.
        chain_type: Type of document combining chain to use. Should be one of "stuff",
            "map_reduce", "map_rerank", and "refine".
        verbose: Whether chains should be run in verbose mode or not. Note that this
            applies to all chains that make up the final chain.
        callback_manager: Callback manager to use for the chain.

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
    return loader_mapping[chain_type](
        llm, verbose=verbose, callback_manager=callback_manager, **kwargs
    )
