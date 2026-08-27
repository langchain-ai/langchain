"""**Chains** are easily reusable components linked together.

Chains encode a sequence of calls to components like models, document retrievers,
other Chains, etc., and provide a simple interface to this sequence.

The Chain interface makes it easy to create apps that are:

    - **Stateful:** add Memory to any Chain to give it state,
    - **Observable:** pass Callbacks to a Chain to execute additional functionality,
        like logging, outside the main sequence of component calls,
    - **Composable:** combine Chains with other components, including other Chains.
"""

from typing import Any

from langchain_classic._api import create_importer

_module_lookup = {
    "APIChain": "langchain_classic.chains.api.base",
    "OpenAPIEndpointChain": "langchain_community.chains.openapi.chain",
    "AnalyzeDocumentChain": "langchain_classic.chains.combine_documents.base",
    "MapReduceDocumentsChain": "langchain_classic.chains.combine_documents.map_reduce",
    "MapRerankDocumentsChain": "langchain_classic.chains.combine_documents.map_rerank",
    "ReduceDocumentsChain": "langchain_classic.chains.combine_documents.reduce",
    "RefineDocumentsChain": "langchain_classic.chains.combine_documents.refine",
    "StuffDocumentsChain": "langchain_classic.chains.combine_documents.stuff",
    "ConstitutionalChain": "langchain_classic.chains.constitutional_ai.base",
    "ConversationChain": "langchain_classic.chains.conversation.base",
    "ChatVectorDBChain": "langchain_classic.chains.conversational_retrieval.base",
    "ConversationalRetrievalChain": (
        "langchain_classic.chains.conversational_retrieval.base"
    ),
    "generate_example": "langchain_classic.chains.example_generator",
    "FlareChain": "langchain_classic.chains.flare.base",
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
    "create_history_aware_retriever": (
        "langchain_classic.chains.history_aware_retriever"
    ),
    "HypotheticalDocumentEmbedder": "langchain_classic.chains.hyde.base",
    "LLMChain": "langchain_classic.chains.llm",
    "LLMCheckerChain": "langchain_classic.chains.llm_checker.base",
    "LLMMathChain": "langchain_classic.chains.llm_math.base",
    "LLMRequestsChain": "langchain_community.chains.llm_requests",
    "LLMSummarizationCheckerChain": (
        "langchain_classic.chains.llm_summarization_checker.base"
    ),
    "load_chain": "langchain_classic.chains.loading",
    "MapReduceChain": "langchain_classic.chains.mapreduce",
    "OpenAIModerationChain": "langchain_classic.chains.moderation",
    "NatBotChain": "langchain_classic.chains.natbot.base",
    "create_citation_fuzzy_match_chain": "langchain_classic.chains.openai_functions",
    "create_citation_fuzzy_match_runnable": "langchain_classic.chains.openai_functions",
    "create_extraction_chain": "langchain_classic.chains.openai_functions",
    "create_extraction_chain_pydantic": "langchain_classic.chains.openai_functions",
    "create_qa_with_sources_chain": "langchain_classic.chains.openai_functions",
    "create_qa_with_structure_chain": "langchain_classic.chains.openai_functions",
    "create_tagging_chain": "langchain_classic.chains.openai_functions",
    "create_tagging_chain_pydantic": "langchain_classic.chains.openai_functions",
    "QAGenerationChain": "langchain_classic.chains.qa_generation.base",
    "QAWithSourcesChain": "langchain_classic.chains.qa_with_sources.base",
    "RetrievalQAWithSourcesChain": "langchain_classic.chains.qa_with_sources.retrieval",
    "VectorDBQAWithSourcesChain": "langchain_classic.chains.qa_with_sources.vector_db",
    "create_retrieval_chain": "langchain_classic.chains.retrieval",
    "RetrievalQA": "langchain_classic.chains.retrieval_qa.base",
    "VectorDBQA": "langchain_classic.chains.retrieval_qa.base",
    "LLMRouterChain": "langchain_classic.chains.router",
    "MultiPromptChain": "langchain_classic.chains.router",
    "MultiRetrievalQAChain": "langchain_classic.chains.router",
    "MultiRouteChain": "langchain_classic.chains.router",
    "RouterChain": "langchain_classic.chains.router",
    "SequentialChain": "langchain_classic.chains.sequential",
    "SimpleSequentialChain": "langchain_classic.chains.sequential",
    "create_sql_query_chain": "langchain_classic.chains.sql_database.query",
    "create_structured_output_runnable": "langchain_classic.chains.structured_output",
    "load_summarize_chain": "langchain_classic.chains.summarize",
    "TransformChain": "langchain_classic.chains.transform",
}

importer = create_importer(__package__, module_lookup=_module_lookup)


def __getattr__(name: str) -> Any:
    return importer(name)


__all__ = list(_module_lookup.keys())
