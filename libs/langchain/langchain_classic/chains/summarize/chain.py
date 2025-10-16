"""Load summarizing chains."""

from collections.abc import Mapping
from typing import Any, Protocol

from langchain_core.callbacks import Callbacks
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate

from langchain_classic.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain_classic.chains.combine_documents.map_reduce import (
    MapReduceDocumentsChain,
)
from langchain_classic.chains.combine_documents.reduce import ReduceDocumentsChain
from langchain_classic.chains.combine_documents.refine import RefineDocumentsChain
from langchain_classic.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_classic.chains.llm import LLMChain
from langchain_classic.chains.summarize import (
    map_reduce_prompt,
    refine_prompts,
    stuff_prompt,
)


class LoadingCallable(Protocol):
    """Interface for loading the combine documents chain."""

    def __call__(
        self,
        llm: BaseLanguageModel,
        **kwargs: Any,
    ) -> BaseCombineDocumentsChain:
        """Callable to load the combine documents chain."""


def _load_stuff_chain(
    llm: BaseLanguageModel,
    *,
    prompt: BasePromptTemplate = stuff_prompt.PROMPT,
    document_variable_name: str = "text",
    verbose: bool | None = None,
    **kwargs: Any,
) -> StuffDocumentsChain:
    llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=verbose)
    """Load a StuffDocumentsChain for summarization.

    Args:
        llm: Language Model to use in the chain.
        prompt: Prompt template that controls how the documents are formatted and
            passed into the LLM.
        document_variable_name: Variable name in the prompt template where the
            document text will be inserted.
        verbose: Whether to log progress and intermediate steps.
        **kwargs: Additional keyword arguments passed to the StuffDocumentsChain.

    Returns:
        A StuffDocumentsChain that takes in documents, formats them with the
        given prompt, and runs the chain on the provided LLM.
    """
    return StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name=document_variable_name,
        verbose=verbose,
        **kwargs,
    )


def _load_map_reduce_chain(
    llm: BaseLanguageModel,
    *,
    map_prompt: BasePromptTemplate = map_reduce_prompt.PROMPT,
    combine_prompt: BasePromptTemplate = map_reduce_prompt.PROMPT,
    combine_document_variable_name: str = "text",
    map_reduce_document_variable_name: str = "text",
    collapse_prompt: BasePromptTemplate | None = None,
    reduce_llm: BaseLanguageModel | None = None,
    collapse_llm: BaseLanguageModel | None = None,
    verbose: bool | None = None,
    token_max: int = 3000,
    callbacks: Callbacks = None,
    collapse_max_retries: int | None = None,
    **kwargs: Any,
) -> MapReduceDocumentsChain:
    map_chain = LLMChain(
        llm=llm,
        prompt=map_prompt,
        verbose=verbose,
        callbacks=callbacks,
    )
    _reduce_llm = reduce_llm or llm
    reduce_chain = LLMChain(
        llm=_reduce_llm,
        prompt=combine_prompt,
        verbose=verbose,
        callbacks=callbacks,
    )
    """Load a MapReduceDocumentsChain for summarization.

    This chain first applies a "map" step to summarize each document,
    then applies a "reduce" step to combine the summaries into a
    final result. Optionally, a "collapse" step can be used to handle
    long intermediate results.

    Args:
        llm: Language Model to use for map and reduce steps.
        map_prompt: Prompt used to summarize each documnet in the map step.
        combine_prompt: Prompt used to combine summaries in the reduce step.
        combine_document_variable_name: Variable name in the `combine_prompt` where
            the mapped summaries are inserted.
        map_reduce_document_variable_name: Variable name in the `map_prompt`
            where document text is inserted.
        collapse_prompt: Optional prompt used to collapse intermediate summaries
            if they exceed the token limit (`token_max`).
        reduce_llm: Optional separate LLM for the reduce step.
            which uses the same model as the map step.
        collapse_llm: Optional separate LLM for the collapse step.
            which uses the same model as the map step.
        verbose: Whether to log progess and intermediate steps.
        token_max: Token threshold that triggers the collapse step during reduction.
        callbacks: Optional callbacks for logging and tracing.
        collapse_max_retries: Maximum retries for the collapse step if it fails.

        **kwargs: Additional keyword arguments passed to the MapReduceDocumentsChain.

    Returns:
        A MapReduceDocumentsChain that maps each document to a summary,
        then reduces all summaries into a single cohesive result.
    """
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain,
        document_variable_name=combine_document_variable_name,
        verbose=verbose,
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
                callbacks=callbacks,
            ),
            document_variable_name=combine_document_variable_name,
        )
    reduce_documents_chain = ReduceDocumentsChain(
        combine_documents_chain=combine_documents_chain,
        collapse_documents_chain=collapse_chain,
        token_max=token_max,
        verbose=verbose,
        callbacks=callbacks,
        collapse_max_retries=collapse_max_retries,
    )
    return MapReduceDocumentsChain(
        llm_chain=map_chain,
        reduce_documents_chain=reduce_documents_chain,
        document_variable_name=map_reduce_document_variable_name,
        verbose=verbose,
        callbacks=callbacks,
        **kwargs,
    )


def _load_refine_chain(
    llm: BaseLanguageModel,
    *,
    question_prompt: BasePromptTemplate = refine_prompts.PROMPT,
    refine_prompt: BasePromptTemplate = refine_prompts.REFINE_PROMPT,
    document_variable_name: str = "text",
    initial_response_name: str = "existing_answer",
    refine_llm: BaseLanguageModel | None = None,
    verbose: bool | None = None,
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


def load_summarize_chain(
    llm: BaseLanguageModel,
    chain_type: str = "stuff",
    verbose: bool | None = None,  # noqa: FBT001
    **kwargs: Any,
) -> BaseCombineDocumentsChain:
    """Load summarizing chain.

    Args:
        llm: Language Model to use in the chain.
        chain_type: Type of document combining chain to use. Should be one of "stuff",
            "map_reduce", and "refine".
        verbose: Whether chains should be run in verbose mode or not. Note that this
            applies to all chains that make up the final chain.
        **kwargs: Additional keyword arguments.

    Returns:
        A chain to use for summarizing.
    """
    loader_mapping: Mapping[str, LoadingCallable] = {
        "stuff": _load_stuff_chain,
        "map_reduce": _load_map_reduce_chain,
        "refine": _load_refine_chain,
    }
    if chain_type not in loader_mapping:
        msg = (
            f"Got unsupported chain type: {chain_type}. "
            f"Should be one of {loader_mapping.keys()}"
        )
        raise ValueError(msg)
    return loader_mapping[chain_type](llm, verbose=verbose, **kwargs)
