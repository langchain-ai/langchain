"""Load question answering chains."""

from collections.abc import Mapping
from typing import Any, Optional, Protocol

from langchain_core._api import deprecated
from langchain_core.callbacks import BaseCallbackManager, Callbacks
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate

from langchain.chains import ReduceDocumentsChain
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain
from langchain.chains.combine_documents.map_rerank import MapRerankDocumentsChain
from langchain.chains.combine_documents.refine import RefineDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import (
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
    callback_manager: Optional[BaseCallbackManager] = None,
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
    prompt: Optional[BasePromptTemplate] = None,
    document_variable_name: str = "context",
    verbose: Optional[bool] = None,
    callback_manager: Optional[BaseCallbackManager] = None,
    callbacks: Callbacks = None,
    **kwargs: Any,
) -> StuffDocumentsChain:
    _prompt = prompt or stuff_prompt.PROMPT_SELECTOR.get_prompt(llm)
    llm_chain = LLMChain(
        llm=llm,
        prompt=_prompt,
        verbose=verbose,  # type: ignore[arg-type]
        callback_manager=callback_manager,
        callbacks=callbacks,
    )
    # TODO: document prompt
    return StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name=document_variable_name,
        verbose=verbose,  # type: ignore[arg-type]
        callback_manager=callback_manager,
        callbacks=callbacks,
        **kwargs,
    )


def _load_map_reduce_chain(
    llm: BaseLanguageModel,
    *,
    question_prompt: Optional[BasePromptTemplate] = None,
    combine_prompt: Optional[BasePromptTemplate] = None,
    combine_document_variable_name: str = "summaries",
    map_reduce_document_variable_name: str = "context",
    collapse_prompt: Optional[BasePromptTemplate] = None,
    reduce_llm: Optional[BaseLanguageModel] = None,
    collapse_llm: Optional[BaseLanguageModel] = None,
    verbose: Optional[bool] = None,
    callback_manager: Optional[BaseCallbackManager] = None,
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
        verbose=verbose,  # type: ignore[arg-type]
        callback_manager=callback_manager,
        callbacks=callbacks,
    )
    _reduce_llm = reduce_llm or llm
    reduce_chain = LLMChain(
        llm=_reduce_llm,
        prompt=_combine_prompt,
        verbose=verbose,  # type: ignore[arg-type]
        callback_manager=callback_manager,
        callbacks=callbacks,
    )
    # TODO: document prompt
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain,
        document_variable_name=combine_document_variable_name,
        verbose=verbose,  # type: ignore[arg-type]
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
                verbose=verbose,  # type: ignore[arg-type]
                callback_manager=callback_manager,
                callbacks=callbacks,
            ),
            document_variable_name=combine_document_variable_name,
            verbose=verbose,  # type: ignore[arg-type]
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
        verbose=verbose,  # type: ignore[arg-type]
        callback_manager=callback_manager,
        callbacks=callbacks,
        **kwargs,
    )


def _load_refine_chain(
    llm: BaseLanguageModel,
    *,
    question_prompt: Optional[BasePromptTemplate] = None,
    refine_prompt: Optional[BasePromptTemplate] = None,
    document_variable_name: str = "context_str",
    initial_response_name: str = "existing_answer",
    refine_llm: Optional[BaseLanguageModel] = None,
    verbose: Optional[bool] = None,
    callback_manager: Optional[BaseCallbackManager] = None,
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
        verbose=verbose,  # type: ignore[arg-type]
        callback_manager=callback_manager,
        callbacks=callbacks,
    )
    _refine_llm = refine_llm or llm
    refine_chain = LLMChain(
        llm=_refine_llm,
        prompt=_refine_prompt,
        verbose=verbose,  # type: ignore[arg-type]
        callback_manager=callback_manager,
        callbacks=callbacks,
    )
    return RefineDocumentsChain(
        initial_llm_chain=initial_chain,
        refine_llm_chain=refine_chain,
        document_variable_name=document_variable_name,
        initial_response_name=initial_response_name,
        verbose=verbose,  # type: ignore[arg-type]
        callback_manager=callback_manager,
        callbacks=callbacks,
        **kwargs,
    )


@deprecated(
    since="0.2.13",
    removal="1.0",
    message=(
        "This class is deprecated. See the following migration guides for replacements "
        "based on `chain_type`:\n"
        "stuff: https://python.langchain.com/docs/versions/migrating_chains/stuff_docs_chain\n"
        "map_reduce: https://python.langchain.com/docs/versions/migrating_chains/map_reduce_chain\n"
        "refine: https://python.langchain.com/docs/versions/migrating_chains/refine_chain\n"
        "map_rerank: https://python.langchain.com/docs/versions/migrating_chains/map_rerank_docs_chain\n"
        "\nSee also guides on retrieval and question-answering here: "
        "https://python.langchain.com/docs/how_to/#qa-with-rag"
    ),
)
def load_qa_chain(
    llm: BaseLanguageModel,
    chain_type: str = "stuff",
    verbose: Optional[bool] = None,  # noqa: FBT001
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
        msg = (
            f"Got unsupported chain type: {chain_type}. "
            f"Should be one of {loader_mapping.keys()}"
        )
        raise ValueError(msg)
    return loader_mapping[chain_type](
        llm,
        verbose=verbose,
        callback_manager=callback_manager,
        **kwargs,
    )
