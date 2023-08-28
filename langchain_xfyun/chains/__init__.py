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

from langchain_xfyun.chains.api.base import APIChain
from langchain_xfyun.chains.api.openapi.chain import OpenAPIEndpointChain
from langchain_xfyun.chains.combine_documents.base import AnalyzeDocumentChain
from langchain_xfyun.chains.combine_documents.map_reduce import MapReduceDocumentsChain
from langchain_xfyun.chains.combine_documents.map_rerank import MapRerankDocumentsChain
from langchain_xfyun.chains.combine_documents.reduce import ReduceDocumentsChain
from langchain_xfyun.chains.combine_documents.refine import RefineDocumentsChain
from langchain_xfyun.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_xfyun.chains.constitutional_ai.base import ConstitutionalChain
from langchain_xfyun.chains.conversation.base import ConversationChain
from langchain_xfyun.chains.conversational_retrieval.base import (
    ChatVectorDBChain,
    ConversationalRetrievalChain,
)
from langchain_xfyun.chains.example_generator import generate_example
from langchain_xfyun.chains.flare.base import FlareChain
from langchain_xfyun.chains.graph_qa.arangodb import ArangoGraphQAChain
from langchain_xfyun.chains.graph_qa.base import GraphQAChain
from langchain_xfyun.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_xfyun.chains.graph_qa.hugegraph import HugeGraphQAChain
from langchain_xfyun.chains.graph_qa.kuzu import KuzuQAChain
from langchain_xfyun.chains.graph_qa.nebulagraph import NebulaGraphQAChain
from langchain_xfyun.chains.graph_qa.neptune_cypher import NeptuneOpenCypherQAChain
from langchain_xfyun.chains.graph_qa.sparql import GraphSparqlQAChain
from langchain_xfyun.chains.hyde.base import HypotheticalDocumentEmbedder
from langchain_xfyun.chains.llm import LLMChain
from langchain_xfyun.chains.llm_bash.base import LLMBashChain
from langchain_xfyun.chains.llm_checker.base import LLMCheckerChain
from langchain_xfyun.chains.llm_math.base import LLMMathChain
from langchain_xfyun.chains.llm_requests import LLMRequestsChain
from langchain_xfyun.chains.llm_summarization_checker.base import LLMSummarizationCheckerChain
from langchain_xfyun.chains.loading import load_chain
from langchain_xfyun.chains.mapreduce import MapReduceChain
from langchain_xfyun.chains.moderation import OpenAIModerationChain
from langchain_xfyun.chains.natbot.base import NatBotChain
from langchain_xfyun.chains.openai_functions import (
    create_citation_fuzzy_match_chain,
    create_extraction_chain,
    create_extraction_chain_pydantic,
    create_qa_with_sources_chain,
    create_qa_with_structure_chain,
    create_tagging_chain,
    create_tagging_chain_pydantic,
)
from langchain_xfyun.chains.qa_generation.base import QAGenerationChain
from langchain_xfyun.chains.qa_with_sources.base import QAWithSourcesChain
from langchain_xfyun.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain_xfyun.chains.qa_with_sources.vector_db import VectorDBQAWithSourcesChain
from langchain_xfyun.chains.retrieval_qa.base import RetrievalQA, VectorDBQA
from langchain_xfyun.chains.router import (
    LLMRouterChain,
    MultiPromptChain,
    MultiRetrievalQAChain,
    MultiRouteChain,
    RouterChain,
)
from langchain_xfyun.chains.sequential import SequentialChain, SimpleSequentialChain
from langchain_xfyun.chains.sql_database.query import create_sql_query_chain
from langchain_xfyun.chains.transform import TransformChain

__all__ = [
    "APIChain",
    "AnalyzeDocumentChain",
    "ArangoGraphQAChain",
    "ChatVectorDBChain",
    "ConstitutionalChain",
    "ConversationChain",
    "ConversationalRetrievalChain",
    "FlareChain",
    "GraphCypherQAChain",
    "GraphQAChain",
    "GraphSparqlQAChain",
    "HugeGraphQAChain",
    "HypotheticalDocumentEmbedder",
    "KuzuQAChain",
    "LLMBashChain",
    "LLMChain",
    "LLMCheckerChain",
    "LLMMathChain",
    "LLMRequestsChain",
    "LLMRouterChain",
    "LLMSummarizationCheckerChain",
    "MapReduceChain",
    "MapReduceDocumentsChain",
    "MapRerankDocumentsChain",
    "MultiPromptChain",
    "MultiRetrievalQAChain",
    "MultiRouteChain",
    "NatBotChain",
    "NebulaGraphQAChain",
    "NeptuneOpenCypherQAChain",
    "OpenAIModerationChain",
    "OpenAPIEndpointChain",
    "QAGenerationChain",
    "QAWithSourcesChain",
    "ReduceDocumentsChain",
    "RefineDocumentsChain",
    "RetrievalQA",
    "RetrievalQAWithSourcesChain",
    "RouterChain",
    "SequentialChain",
    "SimpleSequentialChain",
    "StuffDocumentsChain",
    "TransformChain",
    "VectorDBQA",
    "VectorDBQAWithSourcesChain",
    "create_citation_fuzzy_match_chain",
    "create_extraction_chain",
    "create_extraction_chain_pydantic",
    "create_qa_with_sources_chain",
    "create_qa_with_structure_chain",
    "create_tagging_chain",
    "create_tagging_chain_pydantic",
    "generate_example",
    "load_chain",
    "create_sql_query_chain",
]
