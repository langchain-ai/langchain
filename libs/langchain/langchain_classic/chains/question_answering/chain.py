"""Load question answering chains."""

from typing import Any, Protocol

from langchain_core.callbacks import BaseCallbackManager, Callbacks
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate

from langchain_classic.chains import ReduceDocumentsChain
from langchain_classic.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain_classic.chains.combine_documents.map_reduce import (
    MapReduceDocumentsChain,
)
from langchain_classic.chains.combine_documents.map_rerank import (
    MapRerankDocumentsChain,
)
from langchain_classic.chains.combine_documents.refine import RefineDocumentsChain
from langchain_classic.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_classic.chains.llm import LLMChain
from langchain_classic.chains.question_answering import (
    map_reduce_prompt,
    refine_prompts,
    stuff_prompt,
)
from langchain_classic.chains.question_answering.map_rerank_prompt import (
    PROMPT as MAP_RERANK_PROMPT,
)


class LoadingCallable(Protocol):
    """Interface for loading the combine documents chain."""

    def __call__(
        self,
        llm: BaseLanguageModel,
        **kwargs: Any,
    ) -> BaseCombineDocumentsChain:
        """Callable to load the combine documents chain."""


def _load_map_rerank_chain(
    llm: BaseLanguageModel,
    *,
    prompt: BasePromptTemplate = MAP_RERANK_PROMPT,
    verbose: bool = False,
    document_variable_name: str = "context",
    rank_key: str = "score",
    answer_key: str = "answer",
    callback_manager: BaseCallbackManager | None = None,
    callbacks: Callbacks = None,
    **kwargs: Any,
) -> MapRerankDocumentsChain:
    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=verbose,
        callback_manager=callback_manager,
        callbacks=callbacks,
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
    *,
    prompt: BasePromptTemplate | None = None,
    document_variable_name: str = "context",
    verbose: bool | None = None,
    callback_manager: BaseCallbackManager | None = None,
    callbacks: Callbacks = None,
    **kwargs: Any,
) -> StuffDocumentsChain:
    _prompt = prompt or stuff_prompt.PROMPT_SELECTOR.get_prompt(llm)
    llm_chain = LLMChain(
        llm=llm,
        prompt=_prompt,
        verbose=verbose,
        callback_manager=callback_manager,
        callbacks=callbacks,
    )
    # TODO: document prompt
    return StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name=document_variable_name,
        verbose=verbose,
        callback_manager=callback_manager,
        callbacks=callbacks,
        **kwargs,
    )


def _load_map_reduce_chain(
    llm: BaseLanguageModel,
    *,
    question_prompt: BasePromptTemplate | None = None,
    combine_prompt: BasePromptTemplate | None = None,
    combine_document_variable_name: str = "summaries",
    map_reduce_document_variable_name: str = "context",
    collapse_prompt: BasePromptTemplate | None = None,
    reduce_llm: BaseLanguageModel | None = None,
    collapse_llm: BaseLanguageModel | None = None,
    verbose: bool | None = None,
    callback_manager: BaseCallbackManager | None = None,
    callbacks: Callbacks = None,
    token_max: int = 3000,
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
        callbacks=callbacks,
    )
    _reduce_llm = reduce_llm or llm
    reduce_chain = LLMChain(
        llm=_reduce_llm,
        prompt=_combine_prompt,
        verbose=verbose,
        callback_manager=callback_manager,
        callbacks=callbacks,
    )
    # TODO: document prompt
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain,
        document_variable_name=combine_document_variable_name,
        verbose=verbose,
        callback_manager=callback_manager,
        callbacks=callbacks,
    )
    if collapse_prompt is None:
        collapse_chain = None
        if collapse_llm is not None:
            msg = (
                "collapse_llm provided, but collapse_prompt was not: please "
                "provide one or stop providing collapse_llm."
            )
            raise ValueError(msg)
    else:
        _collapse_llm = collapse_llm or llm
        collapse_chain = StuffDocumentsChain(
            llm_chain=LLMChain(
                llm=_collapse_llm,
                prompt=collapse_prompt,
                verbose=verbose,
                callback_manager=callback_manager,
                callbacks=callbacks,
            ),
            document_variable_name=combine_document_variable_name,
            verbose=verbose,
            callback_manager=callback_manager,
        )
    reduce_documents_chain = ReduceDocumentsChain(
        combine_documents_chain=combine_documents_chain,
        collapse_documents_chain=collapse_chain,
        token_max=token_max,
        verbose=verbose,
    )
    return MapReduceDocumentsChain(
        llm_chain=map_chain,
        document_variable_name=map_reduce_document_variable_name,
        reduce_documents_chain=reduce_documents_chain,
        verbose=verbose,
        callback_manager=callback_manager,
        callbacks=callbacks,
        **kwargs,
    )


def _load_refine_chain(
    llm: BaseLanguageModel,
    *,
    question_prompt: BasePromptTemplate | None = None,
    refine_prompt: BasePromptTemplate | None = None,
    document_variable_name: str = "context_str",
    initial_response_name: str = "existing_answer",
    refine_llm: BaseLanguageModel | None = None,
    verbose: bool | None = None,
    callback_manager: BaseCallbackManager | None = None,
    callbacks: Callbacks = None,
    **kwargs: Any,
) -> RefineDocumentsChain:
    _question_prompt = (
        question_prompt or refine_prompts.QUESTION_PROMPT_SELECTOR.get_prompt(llm)
    )
    _refine_prompt = refine_prompt or refine_prompts.REFINE_PROMPT_SELECTOR.get_prompt(
        llm,
    )
    initial_chain = LLMChain(
        llm=llm,
        prompt=_question_prompt,
        verbose=verbose,
        callback_manager=callback_manager,
        callbacks=callbacks,
    )
    _refine_llm = refine_llm or llm
    refine_chain = LLMChain(
        llm=_refine_llm,
        prompt=_refine_prompt,
        verbose=verbose,
        callback_manager=callback_manager,
        callbacks=callbacks,
    )
    return RefineDocumentsChain(
        initial_llm_chain=initial_chain,
        refine_llm_chain=refine_chain,
        document_variable_name=document_variable_name,
        initial_response_name=initial_response_name,
        verbose=verbose,
        callback_manager=callback_manager,
        callbacks=callbacks,
        **kwargs,
    )
