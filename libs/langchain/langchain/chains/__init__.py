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

from typing import Any

from langchain._api import create_importer

_module_lookup = {
    "APIChain": "langchain.chains.api.base",
    "OpenAPIEndpointChain": "langchain_community.chains.openapi.chain",
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
    "ArangoGraphQAChain": "langchain_community.chains.graph_qa.arangodb",
    "GraphQAChain": "langchain_community.chains.graph_qa.base",
    "GraphCypherQAChain": "langchain_community.chains.graph_qa.cypher",
    "FalkorDBQAChain": "langchain_community.chains.graph_qa.falkordb",
    "HugeGraphQAChain": "langchain_community.chains.graph_qa.hugegraph",
    "KuzuQAChain": "langchain_community.chains.graph_qa.kuzu",
    "NebulaGraphQAChain": "langchain_community.chains.graph_qa.nebulagraph",
    "NeptuneOpenCypherQAChain": "langchain_community.chains.graph_qa.neptune_cypher",
    "NeptuneSparqlQAChain": "langchain_community.chains.graph_qa.neptune_sparql",
    "OntotextGraphDBQAChain": "langchain_community.chains.graph_qa.ontotext_graphdb",
    "GraphSparqlQAChain": "langchain_community.chains.graph_qa.sparql",
    "create_history_aware_retriever": "langchain.chains.history_aware_retriever",
    "HypotheticalDocumentEmbedder": "langchain.chains.hyde.base",
    "LLMChain": "langchain.chains.llm",
    "LLMCheckerChain": "langchain.chains.llm_checker.base",
    "LLMMathChain": "langchain.chains.llm_math.base",
    "LLMRequestsChain": "langchain_community.chains.llm_requests",
    "LLMSummarizationCheckerChain": "langchain.chains.llm_summarization_checker.base",
    "load_chain": "langchain.chains.loading",
    "MapReduceChain": "langchain.chains.mapreduce",
    "OpenAIModerationChain": "langchain.chains.moderation",
    "NatBotChain": "langchain.chains.natbot.base",
    "create_citation_fuzzy_match_chain": "langchain.chains.openai_functions",
    "create_citation_fuzzy_match_runnable": "langchain.chains.openai_functions",
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

importer = create_importer(__package__, module_lookup=_module_lookup)


def __getattr__(name: str) -> Any:
    return importer(name)


__all__ = list(_module_lookup.keys())
