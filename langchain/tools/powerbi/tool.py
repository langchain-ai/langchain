"""Tools for interacting with a Power BI dataset."""
from typing import Any, Dict, Optional

from pydantic import Field, validator

from langchain.chains.llm import LLMChain
from langchain.tools.base import BaseTool
from langchain.tools.powerbi.prompt import (
    BAD_REQUEST_RESPONSE,
    BAD_REQUEST_RESPONSE_ESCALATED,
    DEFAULT_FEWSHOT_EXAMPLES,
    QUESTION_TO_QUERY,
)
from langchain.utilities.powerbi import PowerBIDataset, json_to_md


class QueryPowerBITool(BaseTool):
    """Tool for querying a Power BI Dataset."""

    name = "query_powerbi"
    description = """
    Input to this tool is a detailed and correct DAX query, output is a result from the dataset.
    If the query is not correct, an error message will be returned.
    If an error is returned with Bad request in it, rewrite the query and try again.
    If an error is returned with Unauthorized in it, do not try again, but tell the user to change their authentication.

    Example Input: "EVALUATE ROW("count", COUNTROWS(table1))"
    """  # noqa: E501
    powerbi: PowerBIDataset = Field(exclude=True)
    session_cache: Dict[str, Any] = Field(default_factory=dict, exclude=True)

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def _check_cache(self, tool_input: str) -> Optional[str]:
        """Check if the input is present in the cache.

        If the value is a bad request, overwrite with the escalated version,
        if not present return None."""
        if tool_input not in self.session_cache:
            return None
        if self.session_cache[tool_input] == BAD_REQUEST_RESPONSE:
            self.session_cache[tool_input] = BAD_REQUEST_RESPONSE_ESCALATED
        return self.session_cache[tool_input]

    def _run(self, tool_input: str) -> str:
        """Execute the query, return the results or an error message."""
        if cache := self._check_cache(tool_input):
            return cache
        try:
            self.session_cache[tool_input] = self.powerbi.run(command=tool_input)
        except Exception as exc:  # pylint: disable=broad-except
            if "bad request" in str(exc).lower():
                self.session_cache[tool_input] = BAD_REQUEST_RESPONSE
            elif "unauthorized" in str(exc).lower():
                self.session_cache[
                    tool_input
                ] = "Unauthorized. Try changing your authentication, do not retry."
            else:
                self.session_cache[tool_input] = str(exc)
            return self.session_cache[tool_input]
        if "results" in self.session_cache[tool_input]:
            self.session_cache[tool_input] = json_to_md(
                self.session_cache[tool_input]["results"][0]["tables"][0]["rows"]
            )
        return self.session_cache[tool_input]

    async def _arun(self, tool_input: str) -> str:
        """Execute the query, return the results or an error message."""
        if cache := self._check_cache(tool_input):
            return cache
        try:
            self.session_cache[tool_input] = await self.powerbi.arun(command=tool_input)
        except Exception as exc:  # pylint: disable=broad-except
            if "bad request" in str(exc).lower():
                self.session_cache[tool_input] = BAD_REQUEST_RESPONSE
            elif "unauthorized" in str(exc).lower():
                self.session_cache[
                    tool_input
                ] = "Unauthorized. Try changing your authentication, do not retry."
            else:
                self.session_cache[tool_input] = str(exc)
            return self.session_cache[tool_input]
        if "results" in self.session_cache[tool_input]:
            self.session_cache[tool_input] = json_to_md(
                self.session_cache[tool_input]["results"][0]["tables"][0]["rows"]
            )
        return self.session_cache[tool_input]


class InfoPowerBITool(BaseTool):
    """Tool for getting metadata about a PowerBI Dataset."""

    name = "schema_powerbi"
    description = """
    Input to this tool is a comma-separated list of tables, output is the schema and sample rows for those tables.
    Be sure that the tables actually exist by calling list_tables_powerbi first!

    Example Input: "table1, table2, table3"
    """  # noqa: E501
    powerbi: PowerBIDataset = Field(exclude=True)

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def _run(self, tool_input: str) -> str:
        """Get the schema for tables in a comma-separated list."""
        return self.powerbi.get_table_info(tool_input.split(", "))

    async def _arun(self, tool_input: str) -> str:
        return await self.powerbi.aget_table_info(tool_input.split(", "))


class ListPowerBITool(BaseTool):
    """Tool for getting tables names."""

    name = "list_tables_powerbi"
    description = "Input is an empty string, output is a comma separated list of tables in the database."  # noqa: E501 # pylint: disable=C0301
    powerbi: PowerBIDataset = Field(exclude=True)

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def _run(self, *args: Any, **kwargs: Any) -> str:
        """Get the names of the tables."""
        return ", ".join(self.powerbi.get_table_names())

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        """Get the names of the tables."""
        return ", ".join(self.powerbi.get_table_names())


class InputToQueryTool(BaseTool):
    """Use an LLM to parse the question to a DAX query."""

    name = "question_to_query_powerbi"
    description = """
    Use this tool to create the DAX query from a question, the input is a fully formed question related to the powerbi dataset. Always use this tool before executing a query with query_powerbi!

    Example Input: "How many records are in table1?"
    """  # noqa: E501
    llm_chain: LLMChain
    powerbi: PowerBIDataset = Field(exclude=True)
    template: str = QUESTION_TO_QUERY
    examples: str = DEFAULT_FEWSHOT_EXAMPLES

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    @validator("llm_chain")
    def validate_llm_chain_input_variables(  # pylint: disable=E0213
        cls, llm_chain: LLMChain
    ) -> LLMChain:
        """Make sure the LLM chain has the correct input variables."""
        if llm_chain.prompt.input_variables != [
            "tool_input",
            "tables",
            "schemas",
            "examples",
        ]:
            raise ValueError(
                "LLM chain for InputToQueryTool must have input variables ['tool_input', 'tables', 'schemas', 'examples']"  # noqa: C0301 E501 # pylint: disable=C0301
            )
        return llm_chain

    def _run(self, tool_input: str) -> str:
        """Use the LLM to check the query."""
        return self.llm_chain.predict(
            tool_input=tool_input,
            tables=self.powerbi.get_table_names(),
            schemas=self.powerbi.get_schemas(),
            examples=self.examples,
        )

    async def _arun(self, tool_input: str) -> str:
        return await self.llm_chain.apredict(
            tool_input=tool_input,
            tables=self.powerbi.get_table_names(),
            schemas=self.powerbi.get_schemas(),
            examples=self.examples,
        )
