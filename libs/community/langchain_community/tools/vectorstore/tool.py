"""Tools for interacting with vectorstores."""

import json
from typing import Any, Dict, Optional

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool
from langchain_core.vectorstores import VectorStore
from pydantic import BaseModel, ConfigDict, Field

from langchain_community.llms.openai import OpenAI


class BaseVectorStoreTool(BaseModel):
    """Base class for tools that use a VectorStore."""

    vectorstore: VectorStore = Field(exclude=True)
    llm: BaseLanguageModel = Field(default_factory=lambda: OpenAI(temperature=0))

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )


def _create_description_from_template(values: Dict[str, Any]) -> Dict[str, Any]:
    values["description"] = values["template"].format(name=values["name"])
    return values


class VectorStoreQATool(BaseVectorStoreTool, BaseTool):  # type: ignore[override]
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
        from langchain.chains.retrieval_qa.base import RetrievalQA

        chain = RetrievalQA.from_chain_type(
            self.llm, retriever=self.vectorstore.as_retriever()
        )
        return chain.invoke(
            {chain.input_key: query},
            config={"callbacks": run_manager.get_child() if run_manager else None},
        )[chain.output_key]

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        from langchain.chains.retrieval_qa.base import RetrievalQA

        chain = RetrievalQA.from_chain_type(
            self.llm, retriever=self.vectorstore.as_retriever()
        )
        return (
            await chain.ainvoke(
                {chain.input_key: query},
                config={"callbacks": run_manager.get_child() if run_manager else None},
            )
        )[chain.output_key]


class VectorStoreQAWithSourcesTool(BaseVectorStoreTool, BaseTool):  # type: ignore[override]
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

        from langchain.chains.qa_with_sources.retrieval import (
            RetrievalQAWithSourcesChain,
        )

        chain = RetrievalQAWithSourcesChain.from_chain_type(
            self.llm, retriever=self.vectorstore.as_retriever()
        )
        return json.dumps(
            chain.invoke(
                {chain.question_key: query},
                return_only_outputs=True,
                config={"callbacks": run_manager.get_child() if run_manager else None},
            )
        )

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        from langchain.chains.qa_with_sources.retrieval import (
            RetrievalQAWithSourcesChain,
        )

        chain = RetrievalQAWithSourcesChain.from_chain_type(
            self.llm, retriever=self.vectorstore.as_retriever()
        )
        return json.dumps(
            await chain.ainvoke(
                {chain.question_key: query},
                return_only_outputs=True,
                config={"callbacks": run_manager.get_child() if run_manager else None},
            )
        )
