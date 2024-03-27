# Pinceone as a Tool
import os
from typing import Optional

from langchain_core.callbacks import (AsyncCallbackManagerForToolRun,
                                      CallbackManagerForToolRun)
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator
from langchain_core.tools import BaseTool
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore


class BasePineconeTool(BaseModel):
    """Base tool for interacting with a pinecone vector database."""

    class _QueryPinconeBaseToolInput(BaseModel):
        query: str = Field(..., description="A natural language question")


class PineconeTool(BasePineconeTool, BaseTool):
    """Tool that queries the Pinecone API."""

    name: str = "pinecone-search"
    description: str = (
        "pinecone-search searches from vector database"
        "This tool is will search answer for user's query from Pincone"
        "Input should be a search query."
    )

    vectorstore: PineconeVectorStore = Field(
        default_factory=lambda: PineconeVectorStore.from_existing_index(
            index_name=os.environ["PINECONE_INDEX"], embedding=OpenAIEmbeddings()
        ),
        description="The Pinecone vector store",
    )

    @classmethod
    def create(cls, **kwargs) -> "PineconeTool":
        return cls()

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        return self.vectorstore.similarity_search(query=query)[0].page_content
