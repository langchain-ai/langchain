"""Tools for interacting with an Apache Cassandra database."""

from __future__ import annotations

import traceback
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Type, Union

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

from langchain_community.utilities.cassandra_database import CassandraDatabase

if TYPE_CHECKING:
    from cassandra.cluster import ResultSet


class BaseCassandraDatabaseTool(BaseModel):
    """Base tool for interacting with an Apache Cassandra database."""

    db: CassandraDatabase = Field(exclude=True)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )


class _QueryCassandraDatabaseToolInput(BaseModel):
    query: str = Field(..., description="A detailed and correct CQL query.")


class QueryCassandraDatabaseTool(BaseCassandraDatabaseTool, BaseTool):  # type: ignore[override, override]
    """Tool for querying an Apache Cassandra database with provided CQL."""

    name: str = "cassandra_db_query"
    description: str = """
    Execute a CQL query against the database and get back the result.
    If the query is not correct, an error message will be returned.
    If an error is returned, rewrite the query, check the query, and try again.
    """
    args_schema: Type[BaseModel] = _QueryCassandraDatabaseToolInput

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Union[str, Sequence[Dict[str, Any]], ResultSet]:
        """Execute the query, return the results or an error message."""
        try:
            return self.db.run(query)
        except Exception as e:
            """Format the error message"""
            return f"Error: {e}\n{traceback.format_exc()}"


class _GetSchemaCassandraDatabaseToolInput(BaseModel):
    keyspace: str = Field(
        ...,
        description=("The name of the keyspace for which to return the schema."),
    )


class GetSchemaCassandraDatabaseTool(BaseCassandraDatabaseTool, BaseTool):  # type: ignore[override, override]
    """Tool for getting the schema of a keyspace in an Apache Cassandra database."""

    name: str = "cassandra_db_schema"
    description: str = """
    Input to this tool is a keyspace name, output is a table description 
    of Apache Cassandra tables.
    If the query is not correct, an error message will be returned.
    If an error is returned, report back to the user that the keyspace 
    doesn't exist and stop.
    """

    args_schema: Type[BaseModel] = _GetSchemaCassandraDatabaseToolInput

    def _run(
        self,
        keyspace: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Get the schema for a keyspace."""
        try:
            tables = self.db.get_keyspace_tables(keyspace)
            return "".join([table.as_markdown() + "\n\n" for table in tables])
        except Exception as e:
            """Format the error message"""
            return f"Error: {e}\n{traceback.format_exc()}"


class _GetTableDataCassandraDatabaseToolInput(BaseModel):
    keyspace: str = Field(
        ...,
        description=("The name of the keyspace containing the table."),
    )
    table: str = Field(
        ...,
        description=("The name of the table for which to return data."),
    )
    predicate: str = Field(
        ...,
        description=("The predicate for the query that uses the primary key."),
    )
    limit: int = Field(
        ...,
        description=("The maximum number of rows to return."),
    )


class GetTableDataCassandraDatabaseTool(BaseCassandraDatabaseTool, BaseTool):  # type: ignore[override, override]
    """
    Tool for getting data from a table in an Apache Cassandra database.
    Use the WHERE clause to specify the predicate for the query that uses the
    primary key. A blank predicate will return all rows. Avoid this if possible.
    Use the limit to specify the number of rows to return. A blank limit will
    return all rows.
    """

    name: str = "cassandra_db_select_table_data"
    description: str = """
    Tool for getting data from a table in an Apache Cassandra database. 
    Use the WHERE clause to specify the predicate for the query that uses the 
    primary key. A blank predicate will return all rows. Avoid this if possible. 
    Use the limit to specify the number of rows to return. A blank limit will 
    return all rows.
    """
    args_schema: Type[BaseModel] = _GetTableDataCassandraDatabaseToolInput

    def _run(
        self,
        keyspace: str,
        table: str,
        predicate: str,
        limit: int,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Get data from a table in a keyspace."""
        try:
            return self.db.get_table_data(keyspace, table, predicate, limit)
        except Exception as e:
            """Format the error message"""
            return f"Error: {e}\n{traceback.format_exc()}"
