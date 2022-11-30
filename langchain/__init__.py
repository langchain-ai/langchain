"""Main entrypoint into package."""

from pathlib import Path

with open(Path(__file__).absolute().parents[0] / "VERSION") as _f:
    __version__ = _f.read().strip()

from langchain.agents import MRKLChain, ReActChain, SelfAskWithSearchChain
from langchain.chains import (
    ConversationChain,
    LLMChain,
    LLMMathChain,
    PythonChain,
    SQLDatabaseChain,
    VectorDBQA,
)
from langchain.docstore import InMemoryDocstore, Wikipedia
from langchain.llms import Cohere, HuggingFaceHub, OpenAI
from langchain.prompts import (
    BasePromptTemplate,
    FewShotPromptTemplate,
    Prompt,
    PromptTemplate,
)
from langchain.serpapi import SerpAPIWrapper
from langchain.sql_database import SQLDatabase
from langchain.vectorstores import FAISS, ElasticVectorSearch

__all__ = [
    "LLMChain",
    "LLMMathChain",
    "PythonChain",
    "SelfAskWithSearchChain",
    "SerpAPIWrapper",
    "SerpAPIWrapper",
    "Cohere",
    "OpenAI",
    "BasePromptTemplate",
    "Prompt",
    "FewShotPromptTemplate",
    "PromptTemplate",
    "ReActChain",
    "Wikipedia",
    "HuggingFaceHub",
    "SQLDatabase",
    "SQLDatabaseChain",
    "FAISS",
    "MRKLChain",
    "VectorDBQA",
    "ElasticVectorSearch",
    "InMemoryDocstore",
    "ConversationChain",
]
