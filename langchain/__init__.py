"""Main entrypoint into package."""

from pathlib import Path

with open(Path(__file__).absolute().parents[0] / "VERSION") as _f:
    __version__ = _f.read().strip()

from langchain.chains import (
    LLMChain,
    LLMMathChain,
    PythonChain,
    SerpAPIChain,
    SQLDatabaseChain,
    VectorDBQA,
)
from langchain.docstore import Wikipedia
from langchain.llms import Cohere, HuggingFaceHub, OpenAI
from langchain.prompts import (
    BasePromptTemplate,
    FewShotPromptTemplate,
    Prompt,
    PromptTemplate,
)
from langchain.smart_chains import MRKLChain, ReActChain, SelfAskWithSearchChain
from langchain.sql_database import SQLDatabase
from langchain.vectorstores import FAISS, ElasticVectorSearch

__all__ = [
    "LLMChain",
    "LLMMathChain",
    "PythonChain",
    "SelfAskWithSearchChain",
    "SerpAPIChain",
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
]
