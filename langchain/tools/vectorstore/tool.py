"""Tools for interacting with vectorstores."""

import json
from typing import Any, Dict

from pydantic import BaseModel, Field, root_validator

from langchain.chains.qa_with_sources.vector_db import VectorDBQAWithSourcesChain
from langchain.chains.vector_db_qa.base import VectorDBQA
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

    template: str = (
        "Useful for when you need to answer questions about {name}. "
        "Input should be a fully formed question."
    )
    description = ""

    @root_validator()
    def create_description_from_template(cls, values: dict) -> dict:
        """Create description from template."""
        return _create_description_from_template(values)

    def _run(self, query: str) -> str:
        """Use the tool."""
        chain = VectorDBQA.from_chain_type(self.llm, vectorstore=self.vectorstore)
        return chain.run(query)

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("VectorDBQATool does not support async")


class VectorStoreQAWithSourcesTool(BaseVectorStoreTool, BaseTool):
    """Tool for the VectorDBQAWithSources chain."""

    template: str = (
        "Useful for when you need to answer questions about {name} and the sources "
        "used to construct the answer. Input should be a fully formed question. "
        "Output is a json serialized dictionary with keys `answer` and `sources`. "
        "Only use this tool if the user explicitly asks for sources."
    )
    description = ""

    @root_validator()
    def create_description_from_template(cls, values: dict) -> dict:
        """Create description from template."""
        return _create_description_from_template(values)

    def _run(self, query: str) -> str:
        """Use the tool."""
        chain = VectorDBQAWithSourcesChain.from_chain_type(
            self.llm, vectorstore=self.vectorstore
        )
        return json.dumps(chain({chain.question_key: query}, return_only_outputs=True))

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("VectorDBQATool does not support async")


# if __name__ == "__main__":
#     from langchain import OpenAI, VectorDBQA
#     from langchain.embeddings.openai import OpenAIEmbeddings
#     from langchain.text_splitter import CharacterTextSplitter
#     from langchain.vectorstores import FAISS
#
#     llm = OpenAI(temperature=0)
#     embeddings = OpenAIEmbeddings()
#     text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
#     from langchain.document_loaders import WebBaseLoader
#
#     loader = WebBaseLoader("https://beta.ruff.rs/docs/faq/")
#     docs = loader.load()
#     ruff_texts = text_splitter.split_documents(docs)
#     ruff_db = FAISS.from_documents(ruff_texts, embeddings, collection_name="ruff")
#
#     tool = VectorStoreQATool(name="Ruff", vectorstore=ruff_db, llm=llm)
#     print(tool.description)
#     print(tool.run("What is Ruff?"))
#
#     tool2 = VectorStoreQAWithSourcesTool(name="Ruff", vectorstore=ruff_db, llm=llm)
#     print(tool2.description)
#     print(tool2.run("What is Ruff?"))
