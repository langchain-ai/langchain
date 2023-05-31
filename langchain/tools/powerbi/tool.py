"""Tools for interacting with a Power BI dataset."""
from typing import Any, Dict, Optional, Tuple

from pydantic import Field, validator

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.chains.llm import LLMChain
from langchain.tools.base import BaseTool
from langchain.tools.powerbi.prompt import (
    BAD_REQUEST_RESPONSE,
    DEFAULT_FEWSHOT_EXAMPLES,
    QUESTION_TO_QUERY,
    RETRY_RESPONSE,
)
from langchain.utilities.powerbi import PowerBIDataset, json_to_md


class QueryPowerBITool(BaseTool):
    """Tool for querying a Power BI Dataset."""

    name = "query_powerbi"
    description = """
    Input to this tool is a detailed question about the dataset, output is a result from the dataset. It will try to answer the question using the dataset, and if it cannot, it will ask for clarification.

    Example Input: "How many rows are in table1?"
    """  # noqa: E501
    llm_chain: LLMChain
    powerbi: PowerBIDataset = Field(exclude=True)
    template: Optional[str] = QUESTION_TO_QUERY
    examples: Optional[str] = DEFAULT_FEWSHOT_EXAMPLES
    session_cache: Dict[str, Any] = Field(default_factory=dict, exclude=True)
    max_iterations: int = 5

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
                "LLM chain for QueryPowerBITool must have input variables ['tool_input', 'tables', 'schemas', 'examples'], found %s",  # noqa: C0301 E501 # pylint: disable=C0301
                llm_chain.prompt.input_variables,
            )
        return llm_chain

    def _check_cache(self, tool_input: str) -> Optional[str]:
        """Check if the input is present in the cache.

        If the value is a bad request, overwrite with the escalated version,
        if not present return None."""
        if tool_input not in self.session_cache:
            return None
        return self.session_cache[tool_input]

    def _run(
        self,
        tool_input: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> str:
        """Execute the query, return the results or an error message."""
        if cache := self._check_cache(tool_input):
            return cache

        try:
            query = self.llm_chain.predict(
                tool_input=tool_input,
                tables=self.powerbi.get_table_names(),
                schemas=self.powerbi.get_schemas(),
                examples=self.examples,
            )
        except Exception as exc:  # pylint: disable=broad-except
            self.session_cache[tool_input] = f"Error on call to LLM: {exc}"
            return self.session_cache[tool_input]
        if query == "I cannot answer this":
            self.session_cache[tool_input] = query
            return self.session_cache[tool_input]
        pbi_result = self.powerbi.run(command=query)
        result, error = self._parse_output(pbi_result)

        iterations = kwargs.get("iterations", 0)
        if error and iterations < self.max_iterations:
            return self._run(
                tool_input=RETRY_RESPONSE.format(
                    tool_input=tool_input, query=query, error=error
                ),
                run_manager=run_manager,
                iterations=iterations + 1,
            )

        self.session_cache[tool_input] = (
            result if result else BAD_REQUEST_RESPONSE.format(error=error)
        )
        return self.session_cache[tool_input]

    async def _arun(
        self,
        tool_input: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> str:
        """Execute the query, return the results or an error message."""
        if cache := self._check_cache(tool_input):
            return cache
        try:
            query = await self.llm_chain.apredict(
                tool_input=tool_input,
                tables=self.powerbi.get_table_names(),
                schemas=self.powerbi.get_schemas(),
                examples=self.examples,
            )
        except Exception as exc:  # pylint: disable=broad-except
            self.session_cache[tool_input] = f"Error on call to LLM: {exc}"
            return self.session_cache[tool_input]

        if query == "I cannot answer this":
            self.session_cache[tool_input] = query
            return self.session_cache[tool_input]
        pbi_result = await self.powerbi.arun(command=query)
        result, error = self._parse_output(pbi_result)

        iterations = kwargs.get("iterations", 0)
        if error and iterations < self.max_iterations:
            return await self._arun(
                tool_input=RETRY_RESPONSE.format(
                    tool_input=tool_input, query=query, error=error
                ),
                run_manager=run_manager,
                iterations=iterations + 1,
            )

        self.session_cache[tool_input] = (
            result if result else BAD_REQUEST_RESPONSE.format(error=error)
        )
        return self.session_cache[tool_input]

    def _parse_output(
        self, pbi_result: Dict[str, Any]
    ) -> Tuple[Optional[str], Optional[str]]:
        """Parse the output of the query to a markdown table."""
        if "results" in pbi_result:
            return json_to_md(pbi_result["results"][0]["tables"][0]["rows"]), None

        if "error" in pbi_result:
            if (
                "pbi.error" in pbi_result["error"]
                and "details" in pbi_result["error"]["pbi.error"]
            ):
                return None, pbi_result["error"]["pbi.error"]["details"][0]["detail"]
            return None, pbi_result["error"]
        return None, "Unknown error"


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

    def _run(
        self,
        tool_input: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Get the schema for tables in a comma-separated list."""
        return self.powerbi.get_table_info(tool_input.split(", "))

    async def _arun(
        self,
        tool_input: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        return await self.powerbi.aget_table_info(tool_input.split(", "))


class ListPowerBITool(BaseTool):
    """Tool for getting tables names."""

    name = "list_tables_powerbi"
    description = "Input is an empty string, output is a comma separated list of tables in the database."  # noqa: E501 # pylint: disable=C0301
    powerbi: PowerBIDataset = Field(exclude=True)

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def _run(
        self,
        tool_input: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Get the names of the tables."""
        return ", ".join(self.powerbi.get_table_names())

    async def _arun(
        self,
        tool_input: Optional[str] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Get the names of the tables."""
        return ", ".join(self.powerbi.get_table_names())
