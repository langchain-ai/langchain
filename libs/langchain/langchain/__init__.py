# ruff: noqa: E402
"""Main entrypoint into package."""
import warnings
from importlib import metadata
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from langchain.schema import BaseCache


try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

verbose: bool = False
debug: bool = False
llm_cache: Optional["BaseCache"] = None


def __getattr__(name: str) -> Any:
    warnings.warn(
        f"Importing {name} from langchain root module is no longer supported."
    )
    if name == "MRKLChain":
        from langchain.agents import MRKLChain

        return MRKLChain
    elif name == "ReActChain":
        from langchain.agents import ReActChain

        return ReActChain
    elif name == "SelfAskWithSearchChain":
        from langchain.agents import SelfAskWithSearchChain

        return SelfAskWithSearchChain
    elif name == "ConversationChain":
        from langchain.chains import ConversationChain

        return ConversationChain
    elif name == "LLMBashChain":
        from langchain.chains import LLMBashChain

        return LLMBashChain
    elif name == "LLMChain":
        from langchain.chains import LLMChain

        return LLMChain
    elif name == "LLMCheckerChain":
        from langchain.chains import LLMCheckerChain

        return LLMCheckerChain
    elif name == "LLMMathChain":
        from langchain.chains import LLMMathChain

        return LLMMathChain
    elif name == "QAWithSourcesChain":
        from langchain.chains import QAWithSourcesChain

        return QAWithSourcesChain
    elif name == "VectorDBQA":
        from langchain.chains import VectorDBQA

        return VectorDBQA
    elif name == "VectorDBQAWithSourcesChain":
        from langchain.chains import VectorDBQAWithSourcesChain

        return VectorDBQAWithSourcesChain
    elif name == "InMemoryDocstore":
        from langchain.docstore import InMemoryDocstore

        return InMemoryDocstore
    elif name == "Wikipedia":
        from langchain.docstore import Wikipedia

        return Wikipedia
    elif name == "Anthropic":
        from langchain.llms import Anthropic

        return Anthropic
    elif name == "Banana":
        from langchain.llms import Banana

        return Banana
    elif name == "CerebriumAI":
        from langchain.llms import CerebriumAI

        return CerebriumAI
    elif name == "Cohere":
        from langchain.llms import Cohere

        return Cohere
    elif name == "ForefrontAI":
        from langchain.llms import ForefrontAI

        return ForefrontAI
    elif name == "GooseAI":
        from langchain.llms import GooseAI

        return GooseAI
    elif name == "HuggingFaceHub":
        from langchain.llms import HuggingFaceHub

        return HuggingFaceHub
    elif name == "HuggingFaceTextGenInference":
        from langchain.llms import HuggingFaceTextGenInference

        return HuggingFaceTextGenInference
    elif name == "LlamaCpp":
        from langchain.llms import LlamaCpp

        return LlamaCpp
    elif name == "Modal":
        from langchain.llms import Modal

        return Modal
    elif name == "OpenAI":
        from langchain.llms import OpenAI

        return OpenAI
    elif name == "Petals":
        from langchain.llms import Petals

        return Petals
    elif name == "PipelineAI":
        from langchain.llms import PipelineAI

        return PipelineAI
    elif name == "SagemakerEndpoint":
        from langchain.llms import SagemakerEndpoint

        return SagemakerEndpoint
    elif name == "StochasticAI":
        from langchain.llms import StochasticAI

        return StochasticAI
    elif name == "Writer":
        from langchain.llms import Writer

        return Writer
    elif name == "HuggingFacePipeline":
        from langchain.llms.huggingface_pipeline import HuggingFacePipeline

        return HuggingFacePipeline
    elif name == "FewShotPromptTemplate":
        from langchain.prompts import FewShotPromptTemplate

        return FewShotPromptTemplate
    elif name == "Prompt":
        from langchain.prompts import Prompt

        return Prompt
    elif name == "PromptTemplate":
        from langchain.prompts import PromptTemplate

        return PromptTemplate
    elif name == "BasePromptTemplate":
        from langchain.schema.prompt_template import BasePromptTemplate

        return BasePromptTemplate
    elif name == "ArxivAPIWrapper":
        from langchain.utilities import ArxivAPIWrapper

        return ArxivAPIWrapper
    elif name == "GoldenQueryAPIWrapper":
        from langchain.utilities import GoldenQueryAPIWrapper

        return GoldenQueryAPIWrapper
    elif name == "GoogleSearchAPIWrapper":
        from langchain.utilities import GoogleSearchAPIWrapper

        return GoogleSearchAPIWrapper
    elif name == "GoogleSerperAPIWrapper":
        from langchain.utilities import GoogleSerperAPIWrapper

        return GoogleSerperAPIWrapper
    elif name == "PowerBIDataset":
        from langchain.utilities import PowerBIDataset

        return PowerBIDataset
    elif name == "SearxSearchWrapper":
        from langchain.utilities import SearxSearchWrapper

        return SearxSearchWrapper
    elif name == "WikipediaAPIWrapper":
        from langchain.utilities import WikipediaAPIWrapper

        return WikipediaAPIWrapper
    elif name == "WolframAlphaAPIWrapper":
        from langchain.utilities import WolframAlphaAPIWrapper

        return WolframAlphaAPIWrapper
    elif name == "SQLDatabase":
        from langchain.utilities import SQLDatabase

        return SQLDatabase
    elif name == "FAISS":
        from langchain.vectorstores import FAISS

        return FAISS
    elif name == "ElasticVectorSearch":
        from langchain.vectorstores import ElasticVectorSearch

        return ElasticVectorSearch
    # For backwards compatibility
    elif name == "SerpAPIChain":
        from langchain.utilities import SerpAPIWrapper

        return SerpAPIWrapper
    else:
        raise AttributeError(f"Could not find: {name}")


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
