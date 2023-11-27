"""Toolkit for interacting with a Mongo database."""
from typing import List

from langchain_core.language_models import BaseLanguageModel
from langchain_core.pydantic_v1 import Field

from langchain.agents.agent_toolkits.base import BaseToolkit
from langchain.tools import BaseTool
from langchain.tools.mongo_database.tool import (
    InfoMongoDBTool,
    ListMongoDBTool,
    QueryMongoDBCheckerTool,
    QueryMongoDBTool,
)
from langchain.utilities.mongo_database import MongoDatabase


class MongoDatabaseToolkit(BaseToolkit):
    llm: BaseLanguageModel = Field(exclude=True)
    db: MongoDatabase = Field(exclude=True)

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        list_mongo_database_tool = ListMongoDBTool(db=self.db)
        info_mongo_database_tool_description = (
            "Input to this tool is a comma-separated list of collections, output is "
            "the schema and sample documents for those collections. "
            "Be sure that the collections actually exist by calling "
            f"{list_mongo_database_tool.name} first! "
            "Example Input: collection1, collection2, collection3"
        )
        info_mongo_database_tool = InfoMongoDBTool(
            db=self.db, description=info_mongo_database_tool_description
        )
        query_mongo_database_tool_description = (
            "Input to this tool is a detailed and correct MongoDB query, output is a "
            "result from the database. If the query is not correct, an error message "
            "will be returned. If an error is returned, rewrite the query, check the "
            "query, and try again. If you encounter an issue with Unknown column "
            f"'xxxx' in 'field list', use {info_mongo_database_tool.name} "
            "to query the correct table fields."
        )
        query_mongo_database_tool = QueryMongoDBTool(
            db=self.db, description=query_mongo_database_tool_description
        )
        query_mongo_checker_tool_description = (
            "Use this tool to double check if your query is correct before executing "
            "it. Always use this tool before executing a query with "
            f"{query_mongo_database_tool.name}."
        )
        query_mongo_checker_tool = QueryMongoDBCheckerTool(
            db=self.db, llm=self.llm, description=query_mongo_checker_tool_description
        )
        return [
            list_mongo_database_tool,
            info_mongo_database_tool,
            query_mongo_database_tool,
            query_mongo_checker_tool,
        ]
