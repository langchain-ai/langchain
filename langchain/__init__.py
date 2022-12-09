"""Main entrypoint into package."""

verbose = False
from langchain.agents import MRKLChain, ReActChain, SelfAskWithSearchChain
from langchain.chains import (
    ConversationChain,
    LLMBashChain,
    LLMChain,
    LLMMathChain,
    PALChain,
    QAWithSourcesChain,
    SQLDatabaseChain,
    VectorDBQA,
    VectorDBQAWithSourcesChain,
)
from langchain.docstore import InMemoryDocstore, Wikipedia
from langchain.llms import Cohere, HuggingFaceHub, OpenAI
from langchain.logger import BaseLogger
from langchain.prompts import (
    BasePromptTemplate,
    FewShotPromptTemplate,
    Prompt,
    PromptTemplate,
)
from langchain.serpapi import SerpAPIChain, SerpAPIWrapper
from langchain.sql_database import SQLDatabase
from langchain.vectorstores import FAISS, ElasticVectorSearch

logger = BaseLogger()

__all__ = [
    "LLMChain",
    "LLMBashChain",
    "LLMMathChain",
    "SelfAskWithSearchChain",
    "SerpAPIWrapper",
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
    "InMemoryDocstore",
    "ConversationChain",
    "VectorDBQAWithSourcesChain",
    "QAWithSourcesChain",
    "PALChain",
]
