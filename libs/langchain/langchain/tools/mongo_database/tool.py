# flake8: noqa
"""Tools for interacting with a MongoDB database."""
from typing import Any, Dict, Optional

from langchain.pydantic_v1 import BaseModel, Extra, Field, root_validator

from langchain.schema.language_model import BaseLanguageModel
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.utilities.mongo_database import MongoDBDatabase
from langchain.tools.base import BaseTool
from langchain.tools.mongo_database.prompt import QUERY_CHECKER


class BaseMongoDBTool(BaseModel):
    """Base tool for interacting with a MongoDB database."""

    db: MongoDBDatabase = Field(exclude=True)

    class Config(BaseTool.Config):
        pass


class QueryMongoDBTool(BaseMongoDBTool, BaseTool):
    """Tool for querying a MongoDB database."""

    name: str = "mongo_db_query"
    description: str = """
    Input to this tool is a detailed and correct MongoDB query, output is a result from the database.
    If the query is not correct, an error message will be returned.
    If an error is returned, rewrite the query, check the query, and try again.
    """

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Execute the query, return the results or an error message."""
        return self.db.run(query)


class InfoMongoDBTool(BaseMongoDBTool, BaseTool):
    """Tool for getting metadata about a MongoDB database."""
    
    name: str = "mongo_db_schema"
    description: str = """
    Input to this tool is a comma-separated list of collections, output is the schema and sample documents for those collections.    

    Example Input: "collection1, collection2, collection3"
    """

    def _run(
        self,
        collection_names: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Get information about specified collections."""
        return self.db.get_document_info(collection_names=collection_names)
    

class ListMongoDBTool(BaseMongoDBTool, BaseTool):
    """Tool for listing collections in a MongoDB database."""

    name: str = "mongo_db_list"
    description: str = """
    Output of this tool is a list of collections in the database.
    """

    def _run(
        self,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Get a list of collections in the database."""
        return self.db.collection_info()


class QueryMongoDBCheckerTool(BaseMongoDBTool, BaseTool):
    """Use an LLM to check if a query is correct"""

    template: str = QUERY_CHECKER
    llm: BaseLanguageModel
    llm_chain: LLMChain = Field(init=False)
    name: str = "mongo_db_query_checker"
    description: str = """
    Use this tool to double check a MongoDB query for common mistakes.
    """

    @root_validator(pre=True)
    def _init_llm_chain(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize the LLM chain."""
        if "llm_chain" not in values:
            values["llm_chain"] = LLMChain(
                llm=values.get("llm"),
                prompt=PromptTemplate(
                    template=QUERY_CHECKER, input_variables=["query"]
                ),
            )

        if values["llm_chain"].prompt.input_variables != ["query"]:
            raise ValueError(
                "LLM chain for QueryCheckerTool must have input variables ['query']"
            )
        
        return values
    
    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the LLM to check the query."""
        return self.llm_chain.predict(
            query=query,
            callbacks=run_manager.get_child() if run_manager else None,
        )

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        return await self.llm_chain.apredict(
            query=query,
            callbacks=run_manager.get_child() if run_manager else None,
        )