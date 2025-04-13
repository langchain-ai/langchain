"""Functionality for loading chains."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

import yaml
from langchain_core._api import deprecated
from langchain_core.prompts.loading import (
    _load_output_parser,
    load_prompt,
    load_prompt_from_config,
)

from langchain.chains import ReduceDocumentsChain
from langchain.chains.api.base import APIChain
from langchain.chains.base import Chain
from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain
from langchain.chains.combine_documents.map_rerank import MapRerankDocumentsChain
from langchain.chains.combine_documents.refine import RefineDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.hyde.base import HypotheticalDocumentEmbedder
from langchain.chains.llm import LLMChain
from langchain.chains.llm_checker.base import LLMCheckerChain
from langchain.chains.llm_math.base import LLMMathChain
from langchain.chains.qa_with_sources.base import QAWithSourcesChain
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.vector_db import VectorDBQAWithSourcesChain
from langchain.chains.retrieval_qa.base import RetrievalQA, VectorDBQA

if TYPE_CHECKING:
    from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain

    from langchain.chains.llm_requests import LLMRequestsChain

try:
    from langchain_community.llms.loading import load_llm, load_llm_from_config
except ImportError:

    def load_llm(*args: Any, **kwargs: Any) -> None:  # type: ignore
        raise ImportError(
            "To use this load_llm functionality you must install the "
            "langchain_community package. "
            "You can install it with `pip install langchain_community`"
        )

    def load_llm_from_config(  # type: ignore
        *args: Any, **kwargs: Any
    ) -> None:
        raise ImportError(
            "To use this load_llm_from_config functionality you must install the "
            "langchain_community package. "
            "You can install it with `pip install langchain_community`"
        )


URL_BASE = "https://raw.githubusercontent.com/hwchase17/langchain-hub/master/chains/"


def _load_llm_chain(config: dict, **kwargs: Any) -> LLMChain:
    """Load LLM chain from config dict."""
    if "llm" in config:
        llm_config = config.pop("llm")
        llm = load_llm_from_config(llm_config, **kwargs)
    elif "llm_path" in config:
        llm = load_llm(config.pop("llm_path"), **kwargs)
    else:
        raise ValueError("One of `llm` or `llm_path` must be present.")

    if "prompt" in config:
        prompt_config = config.pop("prompt")
        prompt = load_prompt_from_config(prompt_config)
    elif "prompt_path" in config:
        prompt = load_prompt(config.pop("prompt_path"))
    else:
        raise ValueError("One of `prompt` or `prompt_path` must be present.")
    _load_output_parser(config)

    return LLMChain(llm=llm, prompt=prompt, **config)


def _load_hyde_chain(config: dict, **kwargs: Any) -> HypotheticalDocumentEmbedder:
    """Load hypothetical document embedder chain from config dict."""
    if "llm_chain" in config:
        llm_chain_config = config.pop("llm_chain")
        llm_chain = load_chain_from_config(llm_chain_config, **kwargs)
    elif "llm_chain_path" in config:
        llm_chain = load_chain(config.pop("llm_chain_path"), **kwargs)
    else:
        raise ValueError("One of `llm_chain` or `llm_chain_path` must be present.")
    if "embeddings" in kwargs:
        embeddings = kwargs.pop("embeddings")
    else:
        raise ValueError("`embeddings` must be present.")
    return HypotheticalDocumentEmbedder(
        llm_chain=llm_chain,  # type: ignore[arg-type]
        base_embeddings=embeddings,
        **config,  # type: ignore[arg-type]
    )


def _load_stuff_documents_chain(config: dict, **kwargs: Any) -> StuffDocumentsChain:
    if "llm_chain" in config:
        llm_chain_config = config.pop("llm_chain")
        llm_chain = load_chain_from_config(llm_chain_config, **kwargs)
    elif "llm_chain_path" in config:
        llm_chain = load_chain(config.pop("llm_chain_path"), **kwargs)
    else:
        raise ValueError("One of `llm_chain` or `llm_chain_path` must be present.")

    if not isinstance(llm_chain, LLMChain):
        raise ValueError(f"Expected LLMChain, got {llm_chain}")

    if "document_prompt" in config:
        prompt_config = config.pop("document_prompt")
        document_prompt = load_prompt_from_config(prompt_config)
    elif "document_prompt_path" in config:
        document_prompt = load_prompt(config.pop("document_prompt_path"))
    else:
        raise ValueError(
            "One of `document_prompt` or `document_prompt_path` must be present."
        )

    return StuffDocumentsChain(
        llm_chain=llm_chain, document_prompt=document_prompt, **config
    )


def _load_map_reduce_documents_chain(
    config: dict, **kwargs: Any
) -> MapReduceDocumentsChain:
    if "llm_chain" in config:
        llm_chain_config = config.pop("llm_chain")
        llm_chain = load_chain_from_config(llm_chain_config, **kwargs)
    elif "llm_chain_path" in config:
        llm_chain = load_chain(config.pop("llm_chain_path"), **kwargs)
    else:
        raise ValueError("One of `llm_chain` or `llm_chain_path` must be present.")

    if not isinstance(llm_chain, LLMChain):
        raise ValueError(f"Expected LLMChain, got {llm_chain}")

    if "reduce_documents_chain" in config:
        reduce_documents_chain = load_chain_from_config(
            config.pop("reduce_documents_chain"), **kwargs
        )
    elif "reduce_documents_chain_path" in config:
        reduce_documents_chain = load_chain(
            config.pop("reduce_documents_chain_path"), **kwargs
        )
    else:
        reduce_documents_chain = _load_reduce_documents_chain(config, **kwargs)

    return MapReduceDocumentsChain(
        llm_chain=llm_chain,
        reduce_documents_chain=reduce_documents_chain,  # type: ignore[arg-type]
        **config,
    )


def _load_reduce_documents_chain(config: dict, **kwargs: Any) -> ReduceDocumentsChain:  # type: ignore[valid-type]
    combine_documents_chain = None
    collapse_documents_chain = None

    if "combine_documents_chain" in config:
        combine_document_chain_config = config.pop("combine_documents_chain")
        combine_documents_chain = load_chain_from_config(
            combine_document_chain_config, **kwargs
        )
    elif "combine_document_chain" in config:
        combine_document_chain_config = config.pop("combine_document_chain")
        combine_documents_chain = load_chain_from_config(
            combine_document_chain_config, **kwargs
        )
    elif "combine_documents_chain_path" in config:
        combine_documents_chain = load_chain(
            config.pop("combine_documents_chain_path"), **kwargs
        )
    elif "combine_document_chain_path" in config:
        combine_documents_chain = load_chain(
            config.pop("combine_document_chain_path"), **kwargs
        )
    else:
        raise ValueError(
            "One of `combine_documents_chain` or "
            "`combine_documents_chain_path` must be present."
        )

    if "collapse_documents_chain" in config:
        collapse_document_chain_config = config.pop("collapse_documents_chain")
        if collapse_document_chain_config is None:
            collapse_documents_chain = None
        else:
            collapse_documents_chain = load_chain_from_config(
                collapse_document_chain_config, **kwargs
            )
    elif "collapse_documents_chain_path" in config:
        collapse_documents_chain = load_chain(
            config.pop("collapse_documents_chain_path"), **kwargs
        )
    elif "collapse_document_chain" in config:
        collapse_document_chain_config = config.pop("collapse_document_chain")
        if collapse_document_chain_config is None:
            collapse_documents_chain = None
        else:
            collapse_documents_chain = load_chain_from_config(
                collapse_document_chain_config, **kwargs
            )
    elif "collapse_document_chain_path" in config:
        collapse_documents_chain = load_chain(
            config.pop("collapse_document_chain_path"), **kwargs
        )

    return ReduceDocumentsChain(  # type: ignore[misc]
        combine_documents_chain=combine_documents_chain,
        collapse_documents_chain=collapse_documents_chain,
        **config,
    )


def _load_llm_bash_chain(config: dict, **kwargs: Any) -> Any:
    from langchain_experimental.llm_bash.base import LLMBashChain

    llm_chain = None
    if "llm_chain" in config:
        llm_chain_config = config.pop("llm_chain")
        llm_chain = load_chain_from_config(llm_chain_config, **kwargs)
    elif "llm_chain_path" in config:
        llm_chain = load_chain(config.pop("llm_chain_path"), **kwargs)
    # llm attribute is deprecated in favor of llm_chain, here to support old configs
    elif "llm" in config:
        llm_config = config.pop("llm")
        llm = load_llm_from_config(llm_config, **kwargs)
    # llm_path attribute is deprecated in favor of llm_chain_path,
    # its to support old configs
    elif "llm_path" in config:
        llm = load_llm(config.pop("llm_path"), **kwargs)
    else:
        raise ValueError("One of `llm_chain` or `llm_chain_path` must be present.")
    if "prompt" in config:
        prompt_config = config.pop("prompt")
        prompt = load_prompt_from_config(prompt_config)
    elif "prompt_path" in config:
        prompt = load_prompt(config.pop("prompt_path"))
    if llm_chain:
        return LLMBashChain(llm_chain=llm_chain, prompt=prompt, **config)  # type: ignore[arg-type]
    else:
        return LLMBashChain(llm=llm, prompt=prompt, **config)


def _load_llm_checker_chain(config: dict, **kwargs: Any) -> LLMCheckerChain:
    if "llm" in config:
        llm_config = config.pop("llm")
        llm = load_llm_from_config(llm_config, **kwargs)
    elif "llm_path" in config:
        llm = load_llm(config.pop("llm_path"), **kwargs)
    else:
        raise ValueError("One of `llm` or `llm_path` must be present.")
    if "create_draft_answer_prompt" in config:
        create_draft_answer_prompt_config = config.pop("create_draft_answer_prompt")
        create_draft_answer_prompt = load_prompt_from_config(
            create_draft_answer_prompt_config
        )
    elif "create_draft_answer_prompt_path" in config:
        create_draft_answer_prompt = load_prompt(
            config.pop("create_draft_answer_prompt_path")
        )
    if "list_assertions_prompt" in config:
        list_assertions_prompt_config = config.pop("list_assertions_prompt")
        list_assertions_prompt = load_prompt_from_config(list_assertions_prompt_config)
    elif "list_assertions_prompt_path" in config:
        list_assertions_prompt = load_prompt(config.pop("list_assertions_prompt_path"))
    if "check_assertions_prompt" in config:
        check_assertions_prompt_config = config.pop("check_assertions_prompt")
        check_assertions_prompt = load_prompt_from_config(
            check_assertions_prompt_config
        )
    elif "check_assertions_prompt_path" in config:
        check_assertions_prompt = load_prompt(
            config.pop("check_assertions_prompt_path")
        )
    if "revised_answer_prompt" in config:
        revised_answer_prompt_config = config.pop("revised_answer_prompt")
        revised_answer_prompt = load_prompt_from_config(revised_answer_prompt_config)
    elif "revised_answer_prompt_path" in config:
        revised_answer_prompt = load_prompt(config.pop("revised_answer_prompt_path"))
    return LLMCheckerChain(
        llm=llm,
        create_draft_answer_prompt=create_draft_answer_prompt,  # type: ignore[arg-type]
        list_assertions_prompt=list_assertions_prompt,  # type: ignore[arg-type]
        check_assertions_prompt=check_assertions_prompt,  # type: ignore[arg-type]
        revised_answer_prompt=revised_answer_prompt,  # type: ignore[arg-type]
        **config,
    )


def _load_llm_math_chain(config: dict, **kwargs: Any) -> LLMMathChain:
    llm_chain = None
    if "llm_chain" in config:
        llm_chain_config = config.pop("llm_chain")
        llm_chain = load_chain_from_config(llm_chain_config, **kwargs)
    elif "llm_chain_path" in config:
        llm_chain = load_chain(config.pop("llm_chain_path"), **kwargs)
    # llm attribute is deprecated in favor of llm_chain, here to support old configs
    elif "llm" in config:
        llm_config = config.pop("llm")
        llm = load_llm_from_config(llm_config, **kwargs)
    # llm_path attribute is deprecated in favor of llm_chain_path,
    # its to support old configs
    elif "llm_path" in config:
        llm = load_llm(config.pop("llm_path"), **kwargs)
    else:
        raise ValueError("One of `llm_chain` or `llm_chain_path` must be present.")
    if "prompt" in config:
        prompt_config = config.pop("prompt")
        prompt = load_prompt_from_config(prompt_config)
    elif "prompt_path" in config:
        prompt = load_prompt(config.pop("prompt_path"))
    if llm_chain:
        return LLMMathChain(llm_chain=llm_chain, prompt=prompt, **config)  # type: ignore[arg-type]
    else:
        return LLMMathChain(llm=llm, prompt=prompt, **config)


def _load_map_rerank_documents_chain(
    config: dict, **kwargs: Any
) -> MapRerankDocumentsChain:
    if "llm_chain" in config:
        llm_chain_config = config.pop("llm_chain")
        llm_chain = load_chain_from_config(llm_chain_config, **kwargs)
    elif "llm_chain_path" in config:
        llm_chain = load_chain(config.pop("llm_chain_path"), **kwargs)
    else:
        raise ValueError("One of `llm_chain` or `llm_chain_path` must be present.")
    return MapRerankDocumentsChain(llm_chain=llm_chain, **config)  # type: ignore[arg-type]


def _load_pal_chain(config: dict, **kwargs: Any) -> Any:
    from langchain_experimental.pal_chain import PALChain

    if "llm_chain" in config:
        llm_chain_config = config.pop("llm_chain")
        llm_chain = load_chain_from_config(llm_chain_config, **kwargs)
    elif "llm_chain_path" in config:
        llm_chain = load_chain(config.pop("llm_chain_path"), **kwargs)
    else:
        raise ValueError("One of `llm_chain` or `llm_chain_path` must be present.")
    return PALChain(llm_chain=llm_chain, **config)  # type: ignore[arg-type]


def _load_refine_documents_chain(config: dict, **kwargs: Any) -> RefineDocumentsChain:
    if "initial_llm_chain" in config:
        initial_llm_chain_config = config.pop("initial_llm_chain")
        initial_llm_chain = load_chain_from_config(initial_llm_chain_config, **kwargs)
    elif "initial_llm_chain_path" in config:
        initial_llm_chain = load_chain(config.pop("initial_llm_chain_path"), **kwargs)
    else:
        raise ValueError(
            "One of `initial_llm_chain` or `initial_llm_chain_path` must be present."
        )
    if "refine_llm_chain" in config:
        refine_llm_chain_config = config.pop("refine_llm_chain")
        refine_llm_chain = load_chain_from_config(refine_llm_chain_config, **kwargs)
    elif "refine_llm_chain_path" in config:
        refine_llm_chain = load_chain(config.pop("refine_llm_chain_path"), **kwargs)
    else:
        raise ValueError(
            "One of `refine_llm_chain` or `refine_llm_chain_path` must be present."
        )
    if "document_prompt" in config:
        prompt_config = config.pop("document_prompt")
        document_prompt = load_prompt_from_config(prompt_config)
    elif "document_prompt_path" in config:
        document_prompt = load_prompt(config.pop("document_prompt_path"))
    return RefineDocumentsChain(
        initial_llm_chain=initial_llm_chain,  # type: ignore[arg-type]
        refine_llm_chain=refine_llm_chain,  # type: ignore[arg-type]
        document_prompt=document_prompt,
        **config,
    )


def _load_qa_with_sources_chain(config: dict, **kwargs: Any) -> QAWithSourcesChain:
    if "combine_documents_chain" in config:
        combine_documents_chain_config = config.pop("combine_documents_chain")
        combine_documents_chain = load_chain_from_config(
            combine_documents_chain_config, **kwargs
        )
    elif "combine_documents_chain_path" in config:
        combine_documents_chain = load_chain(
            config.pop("combine_documents_chain_path"), **kwargs
        )
    else:
        raise ValueError(
            "One of `combine_documents_chain` or "
            "`combine_documents_chain_path` must be present."
        )
    return QAWithSourcesChain(combine_documents_chain=combine_documents_chain, **config)  # type: ignore[arg-type]


def _load_sql_database_chain(config: dict, **kwargs: Any) -> Any:
    from langchain_experimental.sql import SQLDatabaseChain

    if "database" in kwargs:
        database = kwargs.pop("database")
    else:
        raise ValueError("`database` must be present.")
    if "llm_chain" in config:
        llm_chain_config = config.pop("llm_chain")
        chain = load_chain_from_config(llm_chain_config, **kwargs)
        return SQLDatabaseChain(llm_chain=chain, database=database, **config)  # type: ignore[arg-type]
    if "llm" in config:
        llm_config = config.pop("llm")
        llm = load_llm_from_config(llm_config, **kwargs)
    elif "llm_path" in config:
        llm = load_llm(config.pop("llm_path"), **kwargs)
    else:
        raise ValueError("One of `llm` or `llm_path` must be present.")
    if "prompt" in config:
        prompt_config = config.pop("prompt")
        prompt = load_prompt_from_config(prompt_config)
    else:
        prompt = None

    return SQLDatabaseChain.from_llm(llm, database, prompt=prompt, **config)


def _load_vector_db_qa_with_sources_chain(
    config: dict, **kwargs: Any
) -> VectorDBQAWithSourcesChain:
    if "vectorstore" in kwargs:
        vectorstore = kwargs.pop("vectorstore")
    else:
        raise ValueError("`vectorstore` must be present.")
    if "combine_documents_chain" in config:
        combine_documents_chain_config = config.pop("combine_documents_chain")
        combine_documents_chain = load_chain_from_config(
            combine_documents_chain_config, **kwargs
        )
    elif "combine_documents_chain_path" in config:
        combine_documents_chain = load_chain(
            config.pop("combine_documents_chain_path"), **kwargs
        )
    else:
        raise ValueError(
            "One of `combine_documents_chain` or "
            "`combine_documents_chain_path` must be present."
        )
    return VectorDBQAWithSourcesChain(
        combine_documents_chain=combine_documents_chain,  # type: ignore[arg-type]
        vectorstore=vectorstore,
        **config,
    )


def _load_retrieval_qa(config: dict, **kwargs: Any) -> RetrievalQA:
    if "retriever" in kwargs:
        retriever = kwargs.pop("retriever")
    else:
        raise ValueError("`retriever` must be present.")
    if "combine_documents_chain" in config:
        combine_documents_chain_config = config.pop("combine_documents_chain")
        combine_documents_chain = load_chain_from_config(
            combine_documents_chain_config, **kwargs
        )
    elif "combine_documents_chain_path" in config:
        combine_documents_chain = load_chain(
            config.pop("combine_documents_chain_path"), **kwargs
        )
    else:
        raise ValueError(
            "One of `combine_documents_chain` or "
            "`combine_documents_chain_path` must be present."
        )
    return RetrievalQA(
        combine_documents_chain=combine_documents_chain,  # type: ignore[arg-type]
        retriever=retriever,
        **config,
    )


def _load_retrieval_qa_with_sources_chain(
    config: dict, **kwargs: Any
) -> RetrievalQAWithSourcesChain:
    if "retriever" in kwargs:
        retriever = kwargs.pop("retriever")
    else:
        raise ValueError("`retriever` must be present.")
    if "combine_documents_chain" in config:
        combine_documents_chain_config = config.pop("combine_documents_chain")
        combine_documents_chain = load_chain_from_config(
            combine_documents_chain_config, **kwargs
        )
    elif "combine_documents_chain_path" in config:
        combine_documents_chain = load_chain(
            config.pop("combine_documents_chain_path"), **kwargs
        )
    else:
        raise ValueError(
            "One of `combine_documents_chain` or "
            "`combine_documents_chain_path` must be present."
        )
    return RetrievalQAWithSourcesChain(
        combine_documents_chain=combine_documents_chain,  # type: ignore[arg-type]
        retriever=retriever,
        **config,
    )


def _load_vector_db_qa(config: dict, **kwargs: Any) -> VectorDBQA:
    if "vectorstore" in kwargs:
        vectorstore = kwargs.pop("vectorstore")
    else:
        raise ValueError("`vectorstore` must be present.")
    if "combine_documents_chain" in config:
        combine_documents_chain_config = config.pop("combine_documents_chain")
        combine_documents_chain = load_chain_from_config(
            combine_documents_chain_config, **kwargs
        )
    elif "combine_documents_chain_path" in config:
        combine_documents_chain = load_chain(
            config.pop("combine_documents_chain_path"), **kwargs
        )
    else:
        raise ValueError(
            "One of `combine_documents_chain` or "
            "`combine_documents_chain_path` must be present."
        )
    return VectorDBQA(
        combine_documents_chain=combine_documents_chain,  # type: ignore[arg-type]
        vectorstore=vectorstore,
        **config,
    )


def _load_graph_cypher_chain(config: dict, **kwargs: Any) -> GraphCypherQAChain:
    if "graph" in kwargs:
        graph = kwargs.pop("graph")
    else:
        raise ValueError("`graph` must be present.")
    if "cypher_generation_chain" in config:
        cypher_generation_chain_config = config.pop("cypher_generation_chain")
        cypher_generation_chain = load_chain_from_config(
            cypher_generation_chain_config, **kwargs
        )
    else:
        raise ValueError("`cypher_generation_chain` must be present.")
    if "qa_chain" in config:
        qa_chain_config = config.pop("qa_chain")
        qa_chain = load_chain_from_config(qa_chain_config, **kwargs)
    else:
        raise ValueError("`qa_chain` must be present.")

    try:
        from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
    except ImportError:
        raise ImportError(
            "To use this GraphCypherQAChain functionality you must install the "
            "langchain_community package. "
            "You can install it with `pip install langchain_community`"
        )
    return GraphCypherQAChain(
        graph=graph,
        cypher_generation_chain=cypher_generation_chain,  # type: ignore[arg-type]
        qa_chain=qa_chain,  # type: ignore[arg-type]
        **config,
    )


def _load_api_chain(config: dict, **kwargs: Any) -> APIChain:
    if "api_request_chain" in config:
        api_request_chain_config = config.pop("api_request_chain")
        api_request_chain = load_chain_from_config(api_request_chain_config, **kwargs)
    elif "api_request_chain_path" in config:
        api_request_chain = load_chain(config.pop("api_request_chain_path"))
    else:
        raise ValueError(
            "One of `api_request_chain` or `api_request_chain_path` must be present."
        )
    if "api_answer_chain" in config:
        api_answer_chain_config = config.pop("api_answer_chain")
        api_answer_chain = load_chain_from_config(api_answer_chain_config, **kwargs)
    elif "api_answer_chain_path" in config:
        api_answer_chain = load_chain(config.pop("api_answer_chain_path"), **kwargs)
    else:
        raise ValueError(
            "One of `api_answer_chain` or `api_answer_chain_path` must be present."
        )
    if "requests_wrapper" in kwargs:
        requests_wrapper = kwargs.pop("requests_wrapper")
    else:
        raise ValueError("`requests_wrapper` must be present.")
    return APIChain(
        api_request_chain=api_request_chain,  # type: ignore[arg-type]
        api_answer_chain=api_answer_chain,  # type: ignore[arg-type]
        requests_wrapper=requests_wrapper,
        **config,
    )


def _load_llm_requests_chain(config: dict, **kwargs: Any) -> LLMRequestsChain:
    try:
        from langchain.chains.llm_requests import LLMRequestsChain
    except ImportError:
        raise ImportError(
            "To use this LLMRequestsChain functionality you must install the "
            "langchain package. "
            "You can install it with `pip install langchain`"
        )

    if "llm_chain" in config:
        llm_chain_config = config.pop("llm_chain")
        llm_chain = load_chain_from_config(llm_chain_config, **kwargs)
    elif "llm_chain_path" in config:
        llm_chain = load_chain(config.pop("llm_chain_path"), **kwargs)
    else:
        raise ValueError("One of `llm_chain` or `llm_chain_path` must be present.")
    if "requests_wrapper" in kwargs:
        requests_wrapper = kwargs.pop("requests_wrapper")
        return LLMRequestsChain(
            llm_chain=llm_chain, requests_wrapper=requests_wrapper, **config
        )
    else:
        return LLMRequestsChain(llm_chain=llm_chain, **config)


type_to_loader_dict = {
    "api_chain": _load_api_chain,
    "hyde_chain": _load_hyde_chain,
    "llm_chain": _load_llm_chain,
    "llm_bash_chain": _load_llm_bash_chain,
    "llm_checker_chain": _load_llm_checker_chain,
    "llm_math_chain": _load_llm_math_chain,
    "llm_requests_chain": _load_llm_requests_chain,
    "pal_chain": _load_pal_chain,
    "qa_with_sources_chain": _load_qa_with_sources_chain,
    "stuff_documents_chain": _load_stuff_documents_chain,
    "map_reduce_documents_chain": _load_map_reduce_documents_chain,
    "reduce_documents_chain": _load_reduce_documents_chain,
    "map_rerank_documents_chain": _load_map_rerank_documents_chain,
    "refine_documents_chain": _load_refine_documents_chain,
    "sql_database_chain": _load_sql_database_chain,
    "vector_db_qa_with_sources_chain": _load_vector_db_qa_with_sources_chain,
    "vector_db_qa": _load_vector_db_qa,
    "retrieval_qa": _load_retrieval_qa,
    "retrieval_qa_with_sources_chain": _load_retrieval_qa_with_sources_chain,
    "graph_cypher_chain": _load_graph_cypher_chain,
}


@deprecated(
    since="0.2.13",
    message=(
        "This function is deprecated and will be removed in langchain 1.0. "
        "At that point chains must be imported from their respective modules."
    ),
    removal="1.0",
)
def load_chain_from_config(config: dict, **kwargs: Any) -> Chain:
    """Load chain from Config Dict."""
    if "_type" not in config:
        raise ValueError("Must specify a chain Type in config")
    config_type = config.pop("_type")

    if config_type not in type_to_loader_dict:
        raise ValueError(f"Loading {config_type} chain not supported")

    chain_loader = type_to_loader_dict[config_type]
    return chain_loader(config, **kwargs)


@deprecated(
    since="0.2.13",
    message=(
        "This function is deprecated and will be removed in langchain 1.0. "
        "At that point chains must be imported from their respective modules."
    ),
    removal="1.0",
)
def load_chain(path: Union[str, Path], **kwargs: Any) -> Chain:
    """Unified method for loading a chain from LangChainHub or local fs."""
    if isinstance(path, str) and path.startswith("lc://"):
        raise RuntimeError(
            "Loading from the deprecated github-based Hub is no longer supported. "
            "Please use the new LangChain Hub at https://smith.langchain.com/hub "
            "instead."
        )
    return _load_chain_from_file(path, **kwargs)


def _load_chain_from_file(file: Union[str, Path], **kwargs: Any) -> Chain:
    """Load chain from file."""
    # Convert file to Path object.
    if isinstance(file, str):
        file_path = Path(file)
    else:
        file_path = file
    # Load from either json or yaml.
    if file_path.suffix == ".json":
        with open(file_path) as f:
            config = json.load(f)
    elif file_path.suffix.endswith((".yaml", ".yml")):
        with open(file_path) as f:
            config = yaml.safe_load(f)
    else:
        raise ValueError("File type must be json or yaml")

    # Override default 'verbose' and 'memory' for the chain
    if "verbose" in kwargs:
        config["verbose"] = kwargs.pop("verbose")
    if "memory" in kwargs:
        config["memory"] = kwargs.pop("memory")

    # Load the chain from the config now.
    return load_chain_from_config(config, **kwargs)
