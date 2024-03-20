"""**Chains** are easily reusable components linked together.

Chains encode a sequence of calls to components like models, document retrievers,
other Chains, etc., and provide a simple interface to this sequence.

The Chain interface makes it easy to create apps that are:

    - **Stateful:** add Memory to any Chain to give it state,
    - **Observable:** pass Callbacks to a Chain to execute additional functionality,
      like logging, outside the main sequence of component calls,
    - **Composable:** combine Chains with other components, including other Chains.

**Class hierarchy:**

.. code-block::

    Chain --> <name>Chain  # Examples: LLMChain, MapReduceChain, RouterChain
"""

import importlib
from typing import Any

_module_lookup = {
    "APIChain": "langchain.chains.api.base",
    "OpenAPIEndpointChain": "langchain.chains.api.openapi.chain",
    "AnalyzeDocumentChain": "langchain.chains.combine_documents.base",
    "MapReduceDocumentsChain": "langchain.chains.combine_documents.map_reduce",
    "MapRerankDocumentsChain": "langchain.chains.combine_documents.map_rerank",
    "ReduceDocumentsChain": "langchain.chains.combine_documents.reduce",
    "RefineDocumentsChain": "langchain.chains.combine_documents.refine",
    "StuffDocumentsChain": "langchain.chains.combine_documents.stuff",
    "ConstitutionalChain": "langchain.chains.constitutional_ai.base",
    "ConversationChain": "langchain.chains.conversation.base",
    "ChatVectorDBChain": "langchain.chains.conversational_retrieval.base",
    "ConversationalRetrievalChain": "langchain.chains.conversational_retrieval.base",
    "generate_example": "langchain.chains.example_generator",
    "FlareChain": "langchain.chains.flare.base",
    "create_history_aware_retriever": "langchain.chains.history_aware_retriever",
    "HypotheticalDocumentEmbedder": "langchain.chains.hyde.base",
    "LLMChain": "langchain.chains.llm",
    "LLMCheckerChain": "langchain.chains.llm_checker.base",
    "LLMMathChain": "langchain.chains.llm_math.base",
    "LLMRequestsChain": "langchain.chains.llm_requests",
    "LLMSummarizationCheckerChain": "langchain.chains.llm_summarization_checker.base",
    "load_chain": "langchain.chains.loading",
    "MapReduceChain": "langchain.chains.mapreduce",
    "OpenAIModerationChain": "langchain.chains.moderation",
    "NatBotChain": "langchain.chains.natbot.base",
    "create_citation_fuzzy_match_chain": "langchain.chains.openai_functions",
    "create_extraction_chain": "langchain.chains.openai_functions",
    "create_extraction_chain_pydantic": "langchain.chains.openai_functions",
    "create_qa_with_sources_chain": "langchain.chains.openai_functions",
    "create_qa_with_structure_chain": "langchain.chains.openai_functions",
    "create_tagging_chain": "langchain.chains.openai_functions",
    "create_tagging_chain_pydantic": "langchain.chains.openai_functions",
    "QAGenerationChain": "langchain.chains.qa_generation.base",
    "QAWithSourcesChain": "langchain.chains.qa_with_sources.base",
    "RetrievalQAWithSourcesChain": "langchain.chains.qa_with_sources.retrieval",
    "VectorDBQAWithSourcesChain": "langchain.chains.qa_with_sources.vector_db",
    "create_retrieval_chain": "langchain.chains.retrieval",
    "RetrievalQA": "langchain.chains.retrieval_qa.base",
    "VectorDBQA": "langchain.chains.retrieval_qa.base",
    "LLMRouterChain": "langchain.chains.router",
    "MultiPromptChain": "langchain.chains.router",
    "MultiRetrievalQAChain": "langchain.chains.router",
    "MultiRouteChain": "langchain.chains.router",
    "RouterChain": "langchain.chains.router",
    "SequentialChain": "langchain.chains.sequential",
    "SimpleSequentialChain": "langchain.chains.sequential",
    "create_sql_query_chain": "langchain.chains.sql_database.query",
    "create_structured_output_runnable": "langchain.chains.structured_output",
    "load_summarize_chain": "langchain.chains.summarize",
    "TransformChain": "langchain.chains.transform",
}

DEPRECATED_GRAPH_IMPORTS = [
    "ArangoGraphQAChain",
    "GraphQAChain",
    "GraphCypherQAChain",
    "FalkorDBQAChain",
    "HugeGraphQAChain",
    "KuzuQAChain",
    "NebulaGraphQAChain",
    "NeptuneOpenCypherQAChain",
    "NeptuneSparqlQAChain",
    "OntotextGraphDBQAChain",
    "GraphSparqlQAChain",
]


def __getattr__(name: str) -> Any:
    if name in _module_lookup:
        module = importlib.import_module(_module_lookup[name])
        return getattr(module, name)
    if name in DEPRECATED_GRAPH_IMPORTS:
        raise ImportError(
            f"{name} has been moved to the langchain-community package. "
            f"See https://github.com/langchain-ai/langchain/discussions/19083 for more "
            f"information.\n\nTo use it install langchain-community:\n\n"
            f"`pip install -U langchain-community`\n\n"
            f"then import with:\n\n"
            f"`from langchain_community.chains.graph_qa import {name}`"
            # noqa: #E501
        )
    raise AttributeError(f"module {__name__} has no attribute {name}")


__all__ = list(_module_lookup.keys())
