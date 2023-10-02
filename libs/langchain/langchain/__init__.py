# ruff: noqa: E402
"""Main entrypoint into package."""
import warnings
from importlib import metadata
from typing import TYPE_CHECKING, Any, Optional

from langchain._api.deprecation import surface_langchain_deprecation_warnings

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


def _is_interactive_env() -> bool:
    """Determine if running within IPython or Jupyter."""
    import sys

    return hasattr(sys, "ps2")


def _warn_on_import(name: str) -> None:
    """Warn on import of deprecated module."""
    if _is_interactive_env():
        # No warnings for interactive environments.
        # This is done to avoid polluting the output of interactive environments
        # where users rely on auto-complete and may trigger this warning
        # even if they are not using any deprecated modules
        return
    warnings.warn(
        f"Importing {name} from langchain root module is no longer supported."
    )


# Surfaces Deprecation and Pending Deprecation warnings from langchain.
surface_langchain_deprecation_warnings()


def __getattr__(name: str) -> Any:
    if name == "MRKLChain":
        from langchain.agents import MRKLChain

        _warn_on_import(name)

        return MRKLChain
    elif name == "ReActChain":
        from langchain.agents import ReActChain

        _warn_on_import(name)

        return ReActChain
    elif name == "SelfAskWithSearchChain":
        from langchain.agents import SelfAskWithSearchChain

        _warn_on_import(name)

        return SelfAskWithSearchChain
    elif name == "ConversationChain":
        from langchain.chains import ConversationChain

        _warn_on_import(name)

        return ConversationChain
    elif name == "LLMBashChain":
        from langchain.chains import LLMBashChain

        _warn_on_import(name)

        return LLMBashChain
    elif name == "LLMChain":
        from langchain.chains import LLMChain

        _warn_on_import(name)

        return LLMChain
    elif name == "LLMCheckerChain":
        from langchain.chains import LLMCheckerChain

        _warn_on_import(name)

        return LLMCheckerChain
    elif name == "LLMMathChain":
        from langchain.chains import LLMMathChain

        _warn_on_import(name)

        return LLMMathChain
    elif name == "QAWithSourcesChain":
        from langchain.chains import QAWithSourcesChain

        _warn_on_import(name)

        return QAWithSourcesChain
    elif name == "VectorDBQA":
        from langchain.chains import VectorDBQA

        _warn_on_import(name)

        return VectorDBQA
    elif name == "VectorDBQAWithSourcesChain":
        from langchain.chains import VectorDBQAWithSourcesChain

        _warn_on_import(name)

        return VectorDBQAWithSourcesChain
    elif name == "InMemoryDocstore":
        from langchain.docstore import InMemoryDocstore

        _warn_on_import(name)

        return InMemoryDocstore
    elif name == "Wikipedia":
        from langchain.docstore import Wikipedia

        _warn_on_import(name)

        return Wikipedia
    elif name == "Anthropic":
        from langchain.llms import Anthropic

        _warn_on_import(name)

        return Anthropic
    elif name == "Banana":
        from langchain.llms import Banana

        _warn_on_import(name)

        return Banana
    elif name == "CerebriumAI":
        from langchain.llms import CerebriumAI

        _warn_on_import(name)

        return CerebriumAI
    elif name == "Cohere":
        from langchain.llms import Cohere

        _warn_on_import(name)

        return Cohere
    elif name == "ForefrontAI":
        from langchain.llms import ForefrontAI

        _warn_on_import(name)

        return ForefrontAI
    elif name == "GooseAI":
        from langchain.llms import GooseAI

        _warn_on_import(name)

        return GooseAI
    elif name == "HuggingFaceHub":
        from langchain.llms import HuggingFaceHub

        _warn_on_import(name)

        return HuggingFaceHub
    elif name == "HuggingFaceTextGenInference":
        from langchain.llms import HuggingFaceTextGenInference

        _warn_on_import(name)

        return HuggingFaceTextGenInference
    elif name == "LlamaCpp":
        from langchain.llms import LlamaCpp

        _warn_on_import(name)

        return LlamaCpp
    elif name == "Modal":
        from langchain.llms import Modal

        _warn_on_import(name)

        return Modal
    elif name == "OpenAI":
        from langchain.llms import OpenAI

        _warn_on_import(name)

        return OpenAI
    elif name == "Petals":
        from langchain.llms import Petals

        _warn_on_import(name)

        return Petals
    elif name == "PipelineAI":
        from langchain.llms import PipelineAI

        _warn_on_import(name)

        return PipelineAI
    elif name == "SagemakerEndpoint":
        from langchain.llms import SagemakerEndpoint

        _warn_on_import(name)

        return SagemakerEndpoint
    elif name == "StochasticAI":
        from langchain.llms import StochasticAI

        _warn_on_import(name)

        return StochasticAI
    elif name == "Writer":
        from langchain.llms import Writer

        _warn_on_import(name)

        return Writer
    elif name == "HuggingFacePipeline":
        from langchain.llms.huggingface_pipeline import HuggingFacePipeline

        _warn_on_import(name)

        return HuggingFacePipeline
    elif name == "FewShotPromptTemplate":
        from langchain.prompts import FewShotPromptTemplate

        _warn_on_import(name)

        return FewShotPromptTemplate
    elif name == "Prompt":
        from langchain.prompts import Prompt

        _warn_on_import(name)

        return Prompt
    elif name == "PromptTemplate":
        from langchain.prompts import PromptTemplate

        _warn_on_import(name)

        return PromptTemplate
    elif name == "BasePromptTemplate":
        from langchain.schema.prompt_template import BasePromptTemplate

        _warn_on_import(name)

        return BasePromptTemplate
    elif name == "ArxivAPIWrapper":
        from langchain.utilities import ArxivAPIWrapper

        _warn_on_import(name)

        return ArxivAPIWrapper
    elif name == "GoldenQueryAPIWrapper":
        from langchain.utilities import GoldenQueryAPIWrapper

        _warn_on_import(name)

        return GoldenQueryAPIWrapper
    elif name == "GoogleSearchAPIWrapper":
        from langchain.utilities import GoogleSearchAPIWrapper

        _warn_on_import(name)

        return GoogleSearchAPIWrapper
    elif name == "GoogleSerperAPIWrapper":
        from langchain.utilities import GoogleSerperAPIWrapper

        _warn_on_import(name)

        return GoogleSerperAPIWrapper
    elif name == "PowerBIDataset":
        from langchain.utilities import PowerBIDataset

        _warn_on_import(name)

        return PowerBIDataset
    elif name == "SearxSearchWrapper":
        from langchain.utilities import SearxSearchWrapper

        _warn_on_import(name)

        return SearxSearchWrapper
    elif name == "WikipediaAPIWrapper":
        from langchain.utilities import WikipediaAPIWrapper

        _warn_on_import(name)

        return WikipediaAPIWrapper
    elif name == "WolframAlphaAPIWrapper":
        from langchain.utilities import WolframAlphaAPIWrapper

        _warn_on_import(name)

        return WolframAlphaAPIWrapper
    elif name == "SQLDatabase":
        from langchain.utilities import SQLDatabase

        _warn_on_import(name)

        return SQLDatabase
    elif name == "FAISS":
        from langchain.vectorstores import FAISS

        _warn_on_import(name)

        return FAISS
    elif name == "ElasticVectorSearch":
        from langchain.vectorstores import ElasticVectorSearch

        _warn_on_import(name)

        return ElasticVectorSearch
    # For backwards compatibility
    elif name == "SerpAPIChain" or name == "SerpAPIWrapper":
        from langchain.utilities import SerpAPIWrapper

        _warn_on_import(name)

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
