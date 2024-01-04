"""Tools for interacting with retrievers."""

import json
from typing import Any, Dict, Optional

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
from langchain.llms.openai import OpenAI
from langchain.pydantic_v1 import BaseModel, Field
from langchain.schema import BaseRetriever
from langchain.schema.language_model import BaseLanguageModel
from langchain.tools.base import BaseTool


class BaseRetrieverTool(BaseModel):
    """Base class for tools that use a Retriever."""

    retriever: BaseRetriever = Field(exclude=True)
    llm: BaseLanguageModel = Field(default_factory=lambda: OpenAI(temperature=0))

    class Config(BaseTool.Config):
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True


def _create_description_from_template(values: Dict[str, Any]) -> Dict[str, Any]:
    values["description"] = values["template"].format(name=values["name"])
    return values


class RetrieverQATool(BaseRetrieverTool, BaseTool):
    """Tool for the RetrieverQA chain. To be initialized with name and chain."""

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
        chain = RetrievalQA.from_chain_type(self.llm, retriever=self.retriever)
        return chain.run(
            query, callbacks=run_manager.get_child() if run_manager else None
        )


class RetrieverQAWithSourcesTool(BaseRetrieverTool, BaseTool):
    """Tool for the RetrieverQAWithSources chain."""

    @staticmethod
    def get_description(name: str, description: str) -> str:
        template: str = (
            "Useful for when you need to answer questions about {name} and the sources "
            "used to construct the answer. "
            "Whenever you need information about {description} "
            "you should ALWAYS use this. "
            " Input should be a fully formed question. "
            "Output is a json serialized dictionary with keys `answer` and `sources`."
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
            self.llm, retriever=self.retriever
        )
        return json.dumps(
            chain(
                {chain.question_key: query},
                return_only_outputs=True,
                callbacks=run_manager.get_child() if run_manager else None,
            )
        )
