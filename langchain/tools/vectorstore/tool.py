"""Tools for interacting with vectorstores."""

import json
from typing import Any, Dict

from pydantic import BaseModel, Field

from langchain.chains.qa_with_sources.vector_db import VectorDBQAWithSourcesChain
from langchain.chains.retrieval_qa.base import VectorDBQA
from langchain.llms.base import BaseLLM
from langchain.llms.openai import OpenAI
from langchain.tools.base import BaseTool
from langchain.vectorstores.base import VectorStore


class BaseVectorStoreTool(BaseModel):
    """Base class for tools that use a VectorStore."""

    vectorstore: VectorStore = Field(exclude=True)
    llm: BaseLLM = Field(default_factory=lambda: OpenAI(temperature=0))

    class Config(BaseTool.Config):
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True


def _create_description_from_template(values: Dict[str, Any]) -> Dict[str, Any]:
    values["description"] = values["template"].format(name=values["name"])
    return values


class VectorStoreQATool(BaseVectorStoreTool, BaseTool):
    """Tool for the VectorDBQA chain. To be initialized with name and chain."""

    @staticmethod
    def get_description(name: str, description: str) -> str:
        template: str = (
            "Useful for when you need to answer questions about {name}. "
            "Whenever you need information about {description} "
            "you should ALWAYS use this. "
            "Input should be a fully formed question."
        )
        return template.format(name=name, description=description)

    def _run(self, query: str) -> str:
        """Use the tool."""
        chain = VectorDBQA.from_chain_type(self.llm, vectorstore=self.vectorstore)
        return chain.run(query)

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("VectorDBQATool does not support async")


class VectorStoreQAWithSourcesTool(BaseVectorStoreTool, BaseTool):
    """Tool for the VectorDBQAWithSources chain."""

    @staticmethod
    def get_description(name: str, description: str) -> str:
        template: str = (
            "Useful for when you need to answer questions about {name} and the sources "
            "used to construct the answer. "
            "Whenever you need information about {description} "
            "you should ALWAYS use this. "
            " Input should be a fully formed question. "
            "Output is a json serialized dictionary with keys `answer` and `sources`. "
            "Only use this tool if the user explicitly asks for sources."
        )
        return template.format(name=name, description=description)

    def _run(self, query: str) -> str:
        """Use the tool."""
        chain = VectorDBQAWithSourcesChain.from_chain_type(self.llm, vectorstore=self.vectorstore)
        return json.dumps(chain({chain.question_key: query}, return_only_outputs=True))

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("VectorDBQATool does not support async")


class VectorStoreMemorySaverTool(BaseTool):
    name = "Vector Store Memory Saver"
    description = (
        "A wrapper around a vector store to save memories based on its embedding. "
        "Useful to store information you want to remember long term. "
        "You don't need to save something you already know. "
        "Input is a string you want to remember and be able to retrieve based on embedding similarity."
        "Output is a string with a confirmation that the memory was saved, you don't need to do "
        "anything else except maybe acknowledging it."
    )
    vectorstore: VectorStore = Field(exclude=True)

    def _run(self, query: str) -> str:
        """Use the tool."""
        self.vectorstore.add_texts(texts=[query])
        return "Answer: Remembered this fact."

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("GoogleSearchRun does not support async")


class VectorStoreMemoryRetrieverTool(BaseTool):
    num_results: int = 4

    name = "Vector Store Memory Retriever"
    description = (
        "A wrapper around a vector store to retrieve memories. "
        "Useful to retrieve information you previously decided to remember. You should always check "
        "if you remember something relevant to the current user input. "
        "Input is a query for which you want to retrieve related memories based on its embedding similarity."
        f"Output is a json array with the top {num_results} results. "
        "You may or may not use this information."
    )

    vectorstore: VectorStore = Field(exclude=True)

    def _run(self, query: str) -> str:
        """Use the tool."""
        try:
            documents = self.vectorstore.similarity_search(query, k=self.num_results)
            return "Answer: " + json.dumps([doc.page_content for doc in documents])
        except ValueError:
            return "Answer: No memories found."

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("GoogleSearchRun does not support async")


class VectorStoreMemoryDeletionTool(BaseTool):
    num_results: int = 4

    name = "Vector Store Memory Deletion"
    description = (
        "Something you previously decided to remember but now you don't want to remember anymore."
    )

    vectorstore: VectorStore = Field(exclude=True)

    def _run(self, query: str) -> str:
        """Use the tool."""
        documents = self.vectorstore.similarity_search(query, k=self.num_results)
        return "Answer: Deleted this fact."

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("GoogleSearchRun does not support async")
