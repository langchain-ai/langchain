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

_exports = [
    {"name": "APIChain", "module": "langchain.chains.api.base"},
    {"name": "OpenAPIEndpointChain", "module": "langchain.chains.api.openapi.chain"},
    {
        "name": "AnalyzeDocumentChain",
        "module": "langchain.chains.combine_documents.base",
    },
    {
        "name": "MapReduceDocumentsChain",
        "module": "langchain.chains.combine_documents.map_reduce",
    },
    {
        "name": "MapRerankDocumentsChain",
        "module": "langchain.chains.combine_documents.map_rerank",
    },
    {
        "name": "ReduceDocumentsChain",
        "module": "langchain.chains.combine_documents.reduce",
    },
    {
        "name": "RefineDocumentsChain",
        "module": "langchain.chains.combine_documents.refine",
    },
    {
        "name": "StuffDocumentsChain",
        "module": "langchain.chains.combine_documents.stuff",
    },
    {
        "name": "ConstitutionalChain",
        "module": "langchain.chains.constitutional_ai.base",
    },
    {"name": "ConversationChain", "module": "langchain.chains.conversation.base"},
    {
        "name": "ChatVectorDBChain",
        "module": "langchain.chains.conversational_retrieval.base",
    },
    {
        "name": "ConversationalRetrievalChain",
        "module": "langchain.chains.conversational_retrieval.base",
    },
    {"name": "generate_example", "module": "langchain.chains.example_generator"},
    {"name": "FlareChain", "module": "langchain.chains.flare.base"},
    {"name": "ArangoGraphQAChain", "module": "langchain.chains.graph_qa.arangodb"},
    {"name": "GraphQAChain", "module": "langchain.chains.graph_qa.base"},
    {"name": "GraphCypherQAChain", "module": "langchain.chains.graph_qa.cypher"},
    {"name": "FalkorDBQAChain", "module": "langchain.chains.graph_qa.falkordb"},
    {"name": "HugeGraphQAChain", "module": "langchain.chains.graph_qa.hugegraph"},
    {"name": "KuzuQAChain", "module": "langchain.chains.graph_qa.kuzu"},
    {"name": "NebulaGraphQAChain", "module": "langchain.chains.graph_qa.nebulagraph"},
    {
        "name": "NeptuneOpenCypherQAChain",
        "module": "langchain.chains.graph_qa.neptune_cypher",
    },
    {
        "name": "NeptuneSparqlQAChain",
        "module": "langchain.chains.graph_qa.neptune_sparql",
    },
    {
        "name": "OntotextGraphDBQAChain",
        "module": "langchain.chains.graph_qa.ontotext_graphdb",
    },
    {"name": "GraphSparqlQAChain", "module": "langchain.chains.graph_qa.sparql"},
    {
        "name": "create_history_aware_retriever",
        "module": "langchain.chains.history_aware_retriever",
    },
    {"name": "HypotheticalDocumentEmbedder", "module": "langchain.chains.hyde.base"},
    {"name": "LLMChain", "module": "langchain.chains.llm"},
    {"name": "LLMCheckerChain", "module": "langchain.chains.llm_checker.base"},
    {"name": "LLMMathChain", "module": "langchain.chains.llm_math.base"},
    {"name": "LLMRequestsChain", "module": "langchain.chains.llm_requests"},
    {
        "name": "LLMSummarizationCheckerChain",
        "module": "langchain.chains.llm_summarization_checker.base",
    },
    {"name": "load_chain", "module": "langchain.chains.loading"},
    {"name": "MapReduceChain", "module": "langchain.chains.mapreduce"},
    {"name": "OpenAIModerationChain", "module": "langchain.chains.moderation"},
    {"name": "NatBotChain", "module": "langchain.chains.natbot.base"},
    {
        "name": "create_citation_fuzzy_match_chain",
        "module": "langchain.chains.openai_functions",
    },
    {"name": "create_extraction_chain", "module": "langchain.chains.openai_functions"},
    {
        "name": "create_extraction_chain_pydantic",
        "module": "langchain.chains.openai_functions",
    },
    {
        "name": "create_qa_with_sources_chain",
        "module": "langchain.chains.openai_functions",
    },
    {
        "name": "create_qa_with_structure_chain",
        "module": "langchain.chains.openai_functions",
    },
    {"name": "create_tagging_chain", "module": "langchain.chains.openai_functions"},
    {
        "name": "create_tagging_chain_pydantic",
        "module": "langchain.chains.openai_functions",
    },
    {"name": "QAGenerationChain", "module": "langchain.chains.qa_generation.base"},
    {"name": "QAWithSourcesChain", "module": "langchain.chains.qa_with_sources.base"},
    {
        "name": "RetrievalQAWithSourcesChain",
        "module": "langchain.chains.qa_with_sources.retrieval",
    },
    {
        "name": "VectorDBQAWithSourcesChain",
        "module": "langchain.chains.qa_with_sources.vector_db",
    },
    {"name": "create_retrieval_chain", "module": "langchain.chains.retrieval"},
    {"name": "RetrievalQA", "module": "langchain.chains.retrieval_qa.base"},
    {"name": "VectorDBQA", "module": "langchain.chains.retrieval_qa.base"},
    {"name": "LLMRouterChain", "module": "langchain.chains.router"},
    {"name": "MultiPromptChain", "module": "langchain.chains.router"},
    {"name": "MultiRetrievalQAChain", "module": "langchain.chains.router"},
    {"name": "MultiRouteChain", "module": "langchain.chains.router"},
    {"name": "RouterChain", "module": "langchain.chains.router"},
    {
        "name": "SequentialChain, SimpleSequentialChain",
        "module": "langchain.chains.sequential",
    },
    {"name": "create_sql_query_chain", "module": "langchain.chains.sql_database.query"},
    {
        "name": "create_structured_output_runnable",
        "module": "langchain.chains.structured_output",
    },
    {"name": "load_summarize_chain", "module": "langchain.chains.summarize"},
    {"name": "TransformChain", "module": "langchain.chains.transform"},
]

_module_lookup = {entry["name"]: entry["module"] for entry in _exports}


def __getattr__(name: str) -> Any:
    if name in _module_lookup:
        return importlib.import_module(_module_lookup[name], name)
    raise AttributeError(f"module {__name__} has no attribute {name}")


__all__ = [entry["name"] for entry in _exports]
