"""Tools for interacting with vectorstores."""

import json
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
from langchain.llms.openai import OpenAI
from langchain.schema.language_model import BaseLanguageModel
from langchain.tools.base import BaseTool
from langchain.vectorstores.base import VectorStore


class BaseVectorStoreTool(BaseModel):
    """Base class for tools that use a VectorStore."""

    vectorstore: VectorStore = Field(exclude=True)
    llm: BaseLanguageModel = Field(default_factory=lambda: OpenAI(temperature=0))

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

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        chain = RetrievalQA.from_chain_type(
            self.llm, retriever=self.vectorstore.as_retriever()
        )
        return chain.run(query)

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("VectorStoreQATool does not support async")


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

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        chain = RetrievalQAWithSourcesChain.from_chain_type(
            self.llm, retriever=self.vectorstore.as_retriever()
        )
        return json.dumps(chain({chain.question_key: query}, return_only_outputs=True))

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("VectorStoreQAWithSourcesTool does not support async")
