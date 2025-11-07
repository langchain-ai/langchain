"""Functionality for loading chains."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from langchain_core._api import deprecated
from langchain_core.prompts.loading import (
    _load_output_parser,
    load_prompt,
    load_prompt_from_config,
)

from langchain_classic.chains import ReduceDocumentsChain
from langchain_classic.chains.api.base import APIChain
from langchain_classic.chains.base import Chain
from langchain_classic.chains.combine_documents.map_reduce import (
    MapReduceDocumentsChain,
)
from langchain_classic.chains.combine_documents.map_rerank import (
    MapRerankDocumentsChain,
)
from langchain_classic.chains.combine_documents.refine import RefineDocumentsChain
from langchain_classic.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_classic.chains.hyde.base import HypotheticalDocumentEmbedder
from langchain_classic.chains.llm import LLMChain
from langchain_classic.chains.llm_checker.base import LLMCheckerChain
from langchain_classic.chains.llm_math.base import LLMMathChain
from langchain_classic.chains.qa_with_sources.base import QAWithSourcesChain
from langchain_classic.chains.qa_with_sources.retrieval import (
    RetrievalQAWithSourcesChain,
)
from langchain_classic.chains.qa_with_sources.vector_db import (
    VectorDBQAWithSourcesChain,
)
from langchain_classic.chains.retrieval_qa.base import RetrievalQA, VectorDBQA

if TYPE_CHECKING:
    from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain

    from langchain_classic.chains.llm_requests import LLMRequestsChain

try:
    from langchain_community.llms.loading import load_llm, load_llm_from_config
except ImportError:

    def load_llm(*_: Any, **__: Any) -> None:
        """Import error for load_llm."""
        msg = (
            "To use this load_llm functionality you must install the "
            "langchain_community package. "
            "You can install it with `pip install langchain_community`"
        )
        raise ImportError(msg)

    def load_llm_from_config(*_: Any, **__: Any) -> None:
        """Import error for load_llm_from_config."""
        msg = (
            "To use this load_llm_from_config functionality you must install the "
            "langchain_community package. "
            "You can install it with `pip install langchain_community`"
        )
        raise ImportError(msg)


URL_BASE = "https://raw.githubusercontent.com/hwchase17/langchain-hub/master/chains/"


def _load_llm_chain(config: dict, **kwargs: Any) -> LLMChain:
    """Load LLM chain from config dict."""
    if "llm" in config:
        llm_config = config.pop("llm")
        llm = load_llm_from_config(llm_config, **kwargs)
    elif "llm_path" in config:
        llm = load_llm(config.pop("llm_path"), **kwargs)
    else:
        msg = "One of `llm` or `llm_path` must be present."
        raise ValueError(msg)

    if "prompt" in config:
        prompt_config = config.pop("prompt")
        prompt = load_prompt_from_config(prompt_config)
    elif "prompt_path" in config:
        prompt = load_prompt(config.pop("prompt_path"))
    else:
        msg = "One of `prompt` or `prompt_path` must be present."
        raise ValueError(msg)
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
        msg = "One of `llm_chain` or `llm_chain_path` must be present."
        raise ValueError(msg)
    if "embeddings" in kwargs:
        embeddings = kwargs.pop("embeddings")
    else:
        msg = "`embeddings` must be present."
        raise ValueError(msg)
    return HypotheticalDocumentEmbedder(
        llm_chain=llm_chain,
        base_embeddings=embeddings,
        **config,
    )


def _load_stuff_documents_chain(config: dict, **kwargs: Any) -> StuffDocumentsChain:
    if "llm_chain" in config:
        llm_chain_config = config.pop("llm_chain")
        llm_chain = load_chain_from_config(llm_chain_config, **kwargs)
    elif "llm_chain_path" in config:
        llm_chain = load_chain(config.pop("llm_chain_path"), **kwargs)
    else:
        msg = "One of `llm_chain` or `llm_chain_path` must be present."
        raise ValueError(msg)

    if not isinstance(llm_chain, LLMChain):
        msg = f"Expected LLMChain, got {llm_chain}"
        raise ValueError(msg)  # noqa: TRY004

    if "document_prompt" in config:
        prompt_config = config.pop("document_prompt")
        document_prompt = load_prompt_from_config(prompt_config)
    elif "document_prompt_path" in config:
        document_prompt = load_prompt(config.pop("document_prompt_path"))
    else:
        msg = "One of `document_prompt` or `document_prompt_path` must be present."
        raise ValueError(msg)

    return StuffDocumentsChain(
        llm_chain=llm_chain,
        document_prompt=document_prompt,
        **config,
    )


def _load_map_reduce_documents_chain(
    config: dict,
    **kwargs: Any,
) -> MapReduceDocumentsChain:
    if "llm_chain" in config:
        llm_chain_config = config.pop("llm_chain")
        llm_chain = load_chain_from_config(llm_chain_config, **kwargs)
    elif "llm_chain_path" in config:
        llm_chain = load_chain(config.pop("llm_chain_path"), **kwargs)
    else:
        msg = "One of `llm_chain` or `llm_chain_path` must be present."
        raise ValueError(msg)

    if not isinstance(llm_chain, LLMChain):
        msg = f"Expected LLMChain, got {llm_chain}"
        raise ValueError(msg)  # noqa: TRY004

    if "reduce_documents_chain" in config:
        reduce_documents_chain = load_chain_from_config(
            config.pop("reduce_documents_chain"),
            **kwargs,
        )
    elif "reduce_documents_chain_path" in config:
        reduce_documents_chain = load_chain(
            config.pop("reduce_documents_chain_path"),
            **kwargs,
        )
    else:
        reduce_documents_chain = _load_reduce_documents_chain(config, **kwargs)

    return MapReduceDocumentsChain(
        llm_chain=llm_chain,
        reduce_documents_chain=reduce_documents_chain,
        **config,
    )


def _load_reduce_documents_chain(config: dict, **kwargs: Any) -> ReduceDocumentsChain:
    combine_documents_chain = None
    collapse_documents_chain = None

    if "combine_documents_chain" in config:
        combine_document_chain_config = config.pop("combine_documents_chain")
        combine_documents_chain = load_chain_from_config(
            combine_document_chain_config,
            **kwargs,
        )
    elif "combine_document_chain" in config:
        combine_document_chain_config = config.pop("combine_document_chain")
        combine_documents_chain = load_chain_from_config(
            combine_document_chain_config,
            **kwargs,
        )
    elif "combine_documents_chain_path" in config:
        combine_documents_chain = load_chain(
            config.pop("combine_documents_chain_path"),
            **kwargs,
        )
    elif "combine_document_chain_path" in config:
        combine_documents_chain = load_chain(
            config.pop("combine_document_chain_path"),
            **kwargs,
        )
    else:
        msg = (
            "One of `combine_documents_chain` or "
            "`combine_documents_chain_path` must be present."
        )
        raise ValueError(msg)

    if "collapse_documents_chain" in config:
        collapse_document_chain_config = config.pop("collapse_documents_chain")
        if collapse_document_chain_config is None:
            collapse_documents_chain = None
        else:
            collapse_documents_chain = load_chain_from_config(
                collapse_document_chain_config,
                **kwargs,
            )
    elif "collapse_documents_chain_path" in config:
        collapse_documents_chain = load_chain(
            config.pop("collapse_documents_chain_path"),
            **kwargs,
        )
    elif "collapse_document_chain" in config:
        collapse_document_chain_config = config.pop("collapse_document_chain")
        if collapse_document_chain_config is None:
            collapse_documents_chain = None
        else:
            collapse_documents_chain = load_chain_from_config(
                collapse_document_chain_config,
                **kwargs,
            )
    elif "collapse_document_chain_path" in config:
        collapse_documents_chain = load_chain(
            config.pop("collapse_document_chain_path"),
            **kwargs,
        )

    return ReduceDocumentsChain(
        combine_documents_chain=combine_documents_chain,
        collapse_documents_chain=collapse_documents_chain,
        **config,
    )


def _load_llm_bash_chain(config: dict, **kwargs: Any) -> Any:
    """Load LLM Bash chain from config dict."""
    msg = (
        "LLMBash Chain is not available through LangChain anymore. "
        "The relevant code can be found in langchain_experimental, "
        "but it is not appropriate for production usage due to security "
        "concerns. Please refer to langchain-experimental repository for more details."
    )
    raise NotImplementedError(msg)


def _load_llm_checker_chain(config: dict, **kwargs: Any) -> LLMCheckerChain:
    if "llm" in config:
        llm_config = config.pop("llm")
        llm = load_llm_from_config(llm_config, **kwargs)
    elif "llm_path" in config:
        llm = load_llm(config.pop("llm_path"), **kwargs)
    else:
        msg = "One of `llm` or `llm_path` must be present."
        raise ValueError(msg)
    if "create_draft_answer_prompt" in config:
        create_draft_answer_prompt_config = config.pop("create_draft_answer_prompt")
        create_draft_answer_prompt = load_prompt_from_config(
            create_draft_answer_prompt_config,
        )
    elif "create_draft_answer_prompt_path" in config:
        create_draft_answer_prompt = load_prompt(
            config.pop("create_draft_answer_prompt_path"),
        )
    if "list_assertions_prompt" in config:
        list_assertions_prompt_config = config.pop("list_assertions_prompt")
        list_assertions_prompt = load_prompt_from_config(list_assertions_prompt_config)
    elif "list_assertions_prompt_path" in config:
        list_assertions_prompt = load_prompt(config.pop("list_assertions_prompt_path"))
    if "check_assertions_prompt" in config:
        check_assertions_prompt_config = config.pop("check_assertions_prompt")
        check_assertions_prompt = load_prompt_from_config(
            check_assertions_prompt_config,
        )
    elif "check_assertions_prompt_path" in config:
        check_assertions_prompt = load_prompt(
            config.pop("check_assertions_prompt_path"),
        )
    if "revised_answer_prompt" in config:
        revised_answer_prompt_config = config.pop("revised_answer_prompt")
        revised_answer_prompt = load_prompt_from_config(revised_answer_prompt_config)
    elif "revised_answer_prompt_path" in config:
        revised_answer_prompt = load_prompt(config.pop("revised_answer_prompt_path"))
    return LLMCheckerChain(
        llm=llm,
        create_draft_answer_prompt=create_draft_answer_prompt,
        list_assertions_prompt=list_assertions_prompt,
        check_assertions_prompt=check_assertions_prompt,
        revised_answer_prompt=revised_answer_prompt,
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
        msg = "One of `llm_chain` or `llm_chain_path` must be present."
        raise ValueError(msg)
    if "prompt" in config:
        prompt_config = config.pop("prompt")
        prompt = load_prompt_from_config(prompt_config)
    elif "prompt_path" in config:
        prompt = load_prompt(config.pop("prompt_path"))
    if llm_chain:
        return LLMMathChain(llm_chain=llm_chain, prompt=prompt, **config)
    return LLMMathChain(llm=llm, prompt=prompt, **config)


def _load_map_rerank_documents_chain(
    config: dict,
    **kwargs: Any,
) -> MapRerankDocumentsChain:
    if "llm_chain" in config:
        llm_chain_config = config.pop("llm_chain")
        llm_chain = load_chain_from_config(llm_chain_config, **kwargs)
    elif "llm_chain_path" in config:
        llm_chain = load_chain(config.pop("llm_chain_path"), **kwargs)
    else:
        msg = "One of `llm_chain` or `llm_chain_path` must be present."
        raise ValueError(msg)
    return MapRerankDocumentsChain(llm_chain=llm_chain, **config)


def _load_pal_chain(config: dict, **kwargs: Any) -> Any:
    msg = (
        "PALChain is not available through LangChain anymore. "
        "The relevant code can be found in langchain_experimental, "
        "but it is not appropriate for production usage due to security "
        "concerns. Please refer to langchain-experimental repository for more details."
    )
    raise NotImplementedError(msg)


def _load_refine_documents_chain(config: dict, **kwargs: Any) -> RefineDocumentsChain:
    if "initial_llm_chain" in config:
        initial_llm_chain_config = config.pop("initial_llm_chain")
        initial_llm_chain = load_chain_from_config(initial_llm_chain_config, **kwargs)
    elif "initial_llm_chain_path" in config:
        initial_llm_chain = load_chain(config.pop("initial_llm_chain_path"), **kwargs)
    else:
        msg = "One of `initial_llm_chain` or `initial_llm_chain_path` must be present."
        raise ValueError(msg)
    if "refine_llm_chain" in config:
        refine_llm_chain_config = config.pop("refine_llm_chain")
        refine_llm_chain = load_chain_from_config(refine_llm_chain_config, **kwargs)
    elif "refine_llm_chain_path" in config:
        refine_llm_chain = load_chain(config.pop("refine_llm_chain_path"), **kwargs)
    else:
        msg = "One of `refine_llm_chain` or `refine_llm_chain_path` must be present."
        raise ValueError(msg)
    if "document_prompt" in config:
        prompt_config = config.pop("document_prompt")
        document_prompt = load_prompt_from_config(prompt_config)
    elif "document_prompt_path" in config:
        document_prompt = load_prompt(config.pop("document_prompt_path"))
    return RefineDocumentsChain(
        initial_llm_chain=initial_llm_chain,
        refine_llm_chain=refine_llm_chain,
        document_prompt=document_prompt,
        **config,
    )


def _load_qa_with_sources_chain(config: dict, **kwargs: Any) -> QAWithSourcesChain:
    if "combine_documents_chain" in config:
        combine_documents_chain_config = config.pop("combine_documents_chain")
        combine_documents_chain = load_chain_from_config(
            combine_documents_chain_config,
            **kwargs,
        )
    elif "combine_documents_chain_path" in config:
        combine_documents_chain = load_chain(
            config.pop("combine_documents_chain_path"),
            **kwargs,
        )
    else:
        msg = (
            "One of `combine_documents_chain` or "
            "`combine_documents_chain_path` must be present."
        )
        raise ValueError(msg)
    return QAWithSourcesChain(combine_documents_chain=combine_documents_chain, **config)


def _load_sql_database_chain(config: dict, **kwargs: Any) -> Any:
    """Load SQL Database chain from config dict."""
    msg = (
        "SQLDatabaseChain is not available through LangChain anymore. "
        "The relevant code can be found in langchain_experimental, "
        "but it is not appropriate for production usage due to security "
        "concerns. Please refer to langchain-experimental repository for more details, "
        "or refer to this tutorial for best practices: "
        "https://python.langchain.com/docs/tutorials/sql_qa/"
    )
    raise NotImplementedError(msg)


def _load_vector_db_qa_with_sources_chain(
    config: dict,
    **kwargs: Any,
) -> VectorDBQAWithSourcesChain:
    if "vectorstore" in kwargs:
        vectorstore = kwargs.pop("vectorstore")
    else:
        msg = "`vectorstore` must be present."
        raise ValueError(msg)
    if "combine_documents_chain" in config:
        combine_documents_chain_config = config.pop("combine_documents_chain")
        combine_documents_chain = load_chain_from_config(
            combine_documents_chain_config,
            **kwargs,
        )
    elif "combine_documents_chain_path" in config:
        combine_documents_chain = load_chain(
            config.pop("combine_documents_chain_path"),
            **kwargs,
        )
    else:
        msg = (
            "One of `combine_documents_chain` or "
            "`combine_documents_chain_path` must be present."
        )
        raise ValueError(msg)
    return VectorDBQAWithSourcesChain(
        combine_documents_chain=combine_documents_chain,
        vectorstore=vectorstore,
        **config,
    )


def _load_retrieval_qa(config: dict, **kwargs: Any) -> RetrievalQA:
    if "retriever" in kwargs:
        retriever = kwargs.pop("retriever")
    else:
        msg = "`retriever` must be present."
        raise ValueError(msg)
    if "combine_documents_chain" in config:
        combine_documents_chain_config = config.pop("combine_documents_chain")
        combine_documents_chain = load_chain_from_config(
            combine_documents_chain_config,
            **kwargs,
        )
    elif "combine_documents_chain_path" in config:
        combine_documents_chain = load_chain(
            config.pop("combine_documents_chain_path"),
            **kwargs,
        )
    else:
        msg = (
            "One of `combine_documents_chain` or "
            "`combine_documents_chain_path` must be present."
        )
        raise ValueError(msg)
    return RetrievalQA(
        combine_documents_chain=combine_documents_chain,
        retriever=retriever,
        **config,
    )


def _load_retrieval_qa_with_sources_chain(
    config: dict,
    **kwargs: Any,
) -> RetrievalQAWithSourcesChain:
    if "retriever" in kwargs:
        retriever = kwargs.pop("retriever")
    else:
        msg = "`retriever` must be present."
        raise ValueError(msg)
    if "combine_documents_chain" in config:
        combine_documents_chain_config = config.pop("combine_documents_chain")
        combine_documents_chain = load_chain_from_config(
            combine_documents_chain_config,
            **kwargs,
        )
    elif "combine_documents_chain_path" in config:
        combine_documents_chain = load_chain(
            config.pop("combine_documents_chain_path"),
            **kwargs,
        )
    else:
        msg = (
            "One of `combine_documents_chain` or "
            "`combine_documents_chain_path` must be present."
        )
        raise ValueError(msg)
    return RetrievalQAWithSourcesChain(
        combine_documents_chain=combine_documents_chain,
        retriever=retriever,
        **config,
    )


def _load_vector_db_qa(config: dict, **kwargs: Any) -> VectorDBQA:
    if "vectorstore" in kwargs:
        vectorstore = kwargs.pop("vectorstore")
    else:
        msg = "`vectorstore` must be present."
        raise ValueError(msg)
    if "combine_documents_chain" in config:
        combine_documents_chain_config = config.pop("combine_documents_chain")
        combine_documents_chain = load_chain_from_config(
            combine_documents_chain_config,
            **kwargs,
        )
    elif "combine_documents_chain_path" in config:
        combine_documents_chain = load_chain(
            config.pop("combine_documents_chain_path"),
            **kwargs,
        )
    else:
        msg = (
            "One of `combine_documents_chain` or "
            "`combine_documents_chain_path` must be present."
        )
        raise ValueError(msg)
    return VectorDBQA(
        combine_documents_chain=combine_documents_chain,
        vectorstore=vectorstore,
        **config,
    )


def _load_graph_cypher_chain(config: dict, **kwargs: Any) -> GraphCypherQAChain:
    if "graph" in kwargs:
        graph = kwargs.pop("graph")
    else:
        msg = "`graph` must be present."
        raise ValueError(msg)
    if "cypher_generation_chain" in config:
        cypher_generation_chain_config = config.pop("cypher_generation_chain")
        cypher_generation_chain = load_chain_from_config(
            cypher_generation_chain_config,
            **kwargs,
        )
    else:
        msg = "`cypher_generation_chain` must be present."
        raise ValueError(msg)
    if "qa_chain" in config:
        qa_chain_config = config.pop("qa_chain")
        qa_chain = load_chain_from_config(qa_chain_config, **kwargs)
    else:
        msg = "`qa_chain` must be present."
        raise ValueError(msg)

    try:
        from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
    except ImportError as e:
        msg = (
            "To use this GraphCypherQAChain functionality you must install the "
            "langchain_community package. "
            "You can install it with `pip install langchain_community`"
        )
        raise ImportError(msg) from e
    return GraphCypherQAChain(
        graph=graph,
        cypher_generation_chain=cypher_generation_chain,
        qa_chain=qa_chain,
        **config,
    )


def _load_api_chain(config: dict, **kwargs: Any) -> APIChain:
    if "api_request_chain" in config:
        api_request_chain_config = config.pop("api_request_chain")
        api_request_chain = load_chain_from_config(api_request_chain_config, **kwargs)
    elif "api_request_chain_path" in config:
        api_request_chain = load_chain(config.pop("api_request_chain_path"))
    else:
        msg = "One of `api_request_chain` or `api_request_chain_path` must be present."
        raise ValueError(msg)
    if "api_answer_chain" in config:
        api_answer_chain_config = config.pop("api_answer_chain")
        api_answer_chain = load_chain_from_config(api_answer_chain_config, **kwargs)
    elif "api_answer_chain_path" in config:
        api_answer_chain = load_chain(config.pop("api_answer_chain_path"), **kwargs)
    else:
        msg = "One of `api_answer_chain` or `api_answer_chain_path` must be present."
        raise ValueError(msg)
    if "requests_wrapper" in kwargs:
        requests_wrapper = kwargs.pop("requests_wrapper")
    else:
        msg = "`requests_wrapper` must be present."
        raise ValueError(msg)
    return APIChain(
        api_request_chain=api_request_chain,
        api_answer_chain=api_answer_chain,
        requests_wrapper=requests_wrapper,
        **config,
    )


def _load_llm_requests_chain(config: dict, **kwargs: Any) -> LLMRequestsChain:
    try:
        from langchain_classic.chains.llm_requests import LLMRequestsChain
    except ImportError as e:
        msg = (
            "To use this LLMRequestsChain functionality you must install the "
            "langchain package. "
            "You can install it with `pip install langchain`"
        )
        raise ImportError(msg) from e

    if "llm_chain" in config:
        llm_chain_config = config.pop("llm_chain")
        llm_chain = load_chain_from_config(llm_chain_config, **kwargs)
    elif "llm_chain_path" in config:
        llm_chain = load_chain(config.pop("llm_chain_path"), **kwargs)
    else:
        msg = "One of `llm_chain` or `llm_chain_path` must be present."
        raise ValueError(msg)
    if "requests_wrapper" in kwargs:
        requests_wrapper = kwargs.pop("requests_wrapper")
        return LLMRequestsChain(
            llm_chain=llm_chain,
            requests_wrapper=requests_wrapper,
            **config,
        )
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
        msg = "Must specify a chain Type in config"
        raise ValueError(msg)
    config_type = config.pop("_type")

    if config_type not in type_to_loader_dict:
        msg = f"Loading {config_type} chain not supported"
        raise ValueError(msg)

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
def load_chain(path: str | Path, **kwargs: Any) -> Chain:
    """Unified method for loading a chain from LangChainHub or local fs."""
    if isinstance(path, str) and path.startswith("lc://"):
        msg = (
            "Loading from the deprecated github-based Hub is no longer supported. "
            "Please use the new LangChain Hub at https://smith.langchain.com/hub "
            "instead."
        )
        raise RuntimeError(msg)
    return _load_chain_from_file(path, **kwargs)


def _load_chain_from_file(file: str | Path, **kwargs: Any) -> Chain:
    """Load chain from file."""
    # Convert file to Path object.
    file_path = Path(file) if isinstance(file, str) else file
    # Load from either json or yaml.
    if file_path.suffix == ".json":
        with file_path.open() as f:
            config = json.load(f)
    elif file_path.suffix.endswith((".yaml", ".yml")):
        with file_path.open() as f:
            config = yaml.safe_load(f)
    else:
        msg = "File type must be json or yaml"
        raise ValueError(msg)

    # Override default 'verbose' and 'memory' for the chain
    if "verbose" in kwargs:
        config["verbose"] = kwargs.pop("verbose")
    if "memory" in kwargs:
        config["memory"] = kwargs.pop("memory")

    # Load the chain from the config now.
    return load_chain_from_config(config, **kwargs)
