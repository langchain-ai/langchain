# ruff: noqa: E402
"""Main entrypoint into package."""
from importlib import metadata
from typing import Optional

from langchain_xfyun.agents import MRKLChain, ReActChain, SelfAskWithSearchChain
from langchain_xfyun.cache import BaseCache
from langchain_xfyun.chains import (
    ConversationChain,
    LLMBashChain,
    LLMChain,
    LLMCheckerChain,
    LLMMathChain,
    QAWithSourcesChain,
    VectorDBQA,
    VectorDBQAWithSourcesChain,
)
from langchain_xfyun.docstore import InMemoryDocstore, Wikipedia
from langchain_xfyun.llms import (
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
from langchain_xfyun.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_xfyun.prompts import (
    FewShotPromptTemplate,
    Prompt,
    PromptTemplate,
)
from langchain_xfyun.schema.prompt_template import BasePromptTemplate
from langchain_xfyun.utilities.arxiv import ArxivAPIWrapper
from langchain_xfyun.utilities.golden_query import GoldenQueryAPIWrapper
from langchain_xfyun.utilities.google_search import GoogleSearchAPIWrapper
from langchain_xfyun.utilities.google_serper import GoogleSerperAPIWrapper
from langchain_xfyun.utilities.powerbi import PowerBIDataset
from langchain_xfyun.utilities.searx_search import SearxSearchWrapper
from langchain_xfyun.utilities.serpapi import SerpAPIWrapper
from langchain_xfyun.utilities.sql_database import SQLDatabase
from langchain_xfyun.utilities.wikipedia import WikipediaAPIWrapper
from langchain_xfyun.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from langchain_xfyun.vectorstores import FAISS, ElasticVectorSearch

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
    "GoldenQueryAPIWrapper",
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
    "PowerBIDataset",
    "FAISS",
    "MRKLChain",
    "VectorDBQA",
    "ElasticVectorSearch",
    "InMemoryDocstore",
    "ConversationChain",
    "VectorDBQAWithSourcesChain",
    "QAWithSourcesChain",
    "LlamaCpp",
    "HuggingFaceTextGenInference",
]
