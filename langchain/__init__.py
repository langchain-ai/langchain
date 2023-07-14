"""Main entrypoint into package."""

from importlib import metadata
from typing import Optional

from langchain.agents import MRKLChain, ReActChain, SelfAskWithSearchChain
from langchain.cache import BaseCache
from langchain.chains import (
    ConversationChain,
    LLMBashChain,
    LLMChain,
    LLMCheckerChain,
    LLMMathChain,
    PALChain,
    QAWithSourcesChain,
    SQLDatabaseChain,
    VectorDBQA,
    VectorDBQAWithSourcesChain,
)
from langchain.docstore import InMemoryDocstore, Wikipedia
from langchain.llms import (
    Anthropic,
    Banana,
    CerebriumAI,
    Cohere,
    ForefrontAI,
    GooseAI,
    HuggingFaceHub,
    HuggingFaceTextGenInference,
    LlamaCpp,
    Modal,
    OpenAI,
    Petals,
    PipelineAI,
    SagemakerEndpoint,
    StochasticAI,
    Writer,
)
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import (
    FewShotPromptTemplate,
    Prompt,
    PromptTemplate,
)
from langchain.schema.prompt_template import BasePromptTemplate
from langchain.sql_database import SQLDatabase
from langchain.utilities.arxiv import ArxivAPIWrapper
from langchain.utilities.google_search import GoogleSearchAPIWrapper
from langchain.utilities.google_serper import GoogleSerperAPIWrapper
from langchain.utilities.powerbi import PowerBIDataset
from langchain.utilities.searx_search import SearxSearchWrapper
from langchain.utilities.serpapi import SerpAPIWrapper
from langchain.utilities.wikipedia import WikipediaAPIWrapper
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from langchain.vectorstores import FAISS, ElasticVectorSearch

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

verbose: bool = False
debug: bool = False
llm_cache: Optional[BaseCache] = None

# For backwards compatibility
SerpAPIChain = SerpAPIWrapper

__all__ = [
    "LLMChain",
    "LLMBashChain",
    "LLMCheckerChain",
    "LLMMathChain",
    "ArxivAPIWrapper",
    "SelfAskWithSearchChain",
    "SerpAPIWrapper",
    "SerpAPIChain",
    "SearxSearchWrapper",
    "GoogleSearchAPIWrapper",
    "GoogleSerperAPIWrapper",
    "WolframAlphaAPIWrapper",
    "WikipediaAPIWrapper",
    "Anthropic",
    "Banana",
    "CerebriumAI",
    "Cohere",
    "ForefrontAI",
    "GooseAI",
    "Modal",
    "OpenAI",
    "Petals",
    "PipelineAI",
    "StochasticAI",
    "Writer",
    "BasePromptTemplate",
    "Prompt",
    "FewShotPromptTemplate",
    "PromptTemplate",
    "ReActChain",
    "Wikipedia",
    "HuggingFaceHub",
    "SagemakerEndpoint",
    "HuggingFacePipeline",
    "SQLDatabase",
    "SQLDatabaseChain",
    "PowerBIDataset",
    "FAISS",
    "MRKLChain",
    "VectorDBQA",
    "ElasticVectorSearch",
    "InMemoryDocstore",
    "ConversationChain",
    "VectorDBQAWithSourcesChain",
    "QAWithSourcesChain",
    "PALChain",
    "LlamaCpp",
    "HuggingFaceTextGenInference",
]
