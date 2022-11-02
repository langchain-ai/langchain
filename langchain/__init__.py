"""Main entrypoint into package."""

from pathlib import Path

with open(Path(__file__).absolute().parents[0] / "VERSION") as _f:
    __version__ = _f.read().strip()

from langchain.chains import (
    LLMChain,
    LLMMathChain,
    PythonChain,
    ReActChain,
    SelfAskWithSearchChain,
    SerpAPIChain,
    SQLDatabaseChain,
)
from langchain.docstore import Wikipedia
from langchain.faiss import FAISS
from langchain.llms import Cohere, HuggingFaceHub, OpenAI
from langchain.prompt import Prompt
from langchain.sql_database import SQLDatabase

__all__ = [
    "LLMChain",
    "LLMMathChain",
    "PythonChain",
    "SelfAskWithSearchChain",
    "SerpAPIChain",
    "Cohere",
    "OpenAI",
    "Prompt",
    "ReActChain",
    "Wikipedia",
    "HuggingFaceHub",
    "SQLDatabase",
    "SQLDatabaseChain",
    "FAISS",
]
