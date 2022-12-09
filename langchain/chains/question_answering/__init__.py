"""Load question answering chains."""
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain
from langchain.chains.combine_documents.refine import RefineDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import (
    map_reduce_prompt,
    refine_prompt,
    stuff_prompt,
)
from langchain.llms.base import LLM


def _load_stuff_chain(llm: LLM) -> StuffDocumentsChain:
    llm_chain = LLMChain(llm=llm, prompt=stuff_prompt.PROMPT)
    # TODO: document prompt
    return StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="context")


def _load_map_reduce_chain(llm: LLM) -> MapReduceDocumentsChain:
    map_chain = LLMChain(llm=llm, prompt=map_reduce_prompt.QUESTION_PROMPT)
    reduce_chain = LLMChain(llm=llm, prompt=map_reduce_prompt.COMBINE_PROMPT)
    # TODO: document prompt
    combine_document_chain = StuffDocumentsChain(
        llm_chain=reduce_chain, document_variable_name="summaries"
    )
    return MapReduceDocumentsChain(
        llm_chain=map_chain,
        combine_document_chain=combine_document_chain,
        document_variable_name="context",
    )


def _load_refine_chain(llm: LLM) -> RefineDocumentsChain:
    initial_chain = LLMChain(llm=llm, prompt=refine_prompt.DEFAULT_TEXT_QA_PROMPT)
    refine_chain = LLMChain(llm=llm, prompt=refine_prompt.DEFAULT_REFINE_PROMPT)
    return RefineDocumentsChain(
        initial_llm_chain=initial_chain,
        refine_llm_chain=refine_chain,
        document_variable_name="context_str",
        initial_response_name="existing_answer",
    )


def load_qa_chain(llm: LLM, chain_type: str = "stuff") -> BaseCombineDocumentsChain:
    """Load question answering chain.

    Args:
        llm: Language Model to use in the chain.
        chain_type: Type of document combining chain to use. Should be one of "stuff",
            "map_reduce", and "refine".

    Returns:
        A chain to use for question answering.
    """
    loader_mapping = {
        "stuff": _load_stuff_chain,
        "map_reduce": _load_map_reduce_chain,
        "refine": _load_refine_chain,
    }
    if chain_type not in loader_mapping:
        raise ValueError(
            f"Got unsupported chain type: {chain_type}. "
            f"Should be one of {loader_mapping.keys()}"
        )
    return loader_mapping[chain_type](llm)
