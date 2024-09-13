"""Tools for interacting with a Power BI dataset."""

import logging
from time import perf_counter
from typing import Any, Dict, Optional, Tuple

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import ConfigDict, Field, model_validator

from langchain_community.chat_models.openai import _import_tiktoken
from langchain_community.tools.powerbi.prompt import (
    BAD_REQUEST_RESPONSE,
    DEFAULT_FEWSHOT_EXAMPLES,
    RETRY_RESPONSE,
)
from langchain_community.utilities.powerbi import PowerBIDataset, json_to_md

logger = logging.getLogger(__name__)


class QueryPowerBITool(BaseTool):
    """Tool for querying a Power BI Dataset."""

    name: str = "query_powerbi"
    description: str = """
    Input to this tool is a detailed question about the dataset, output is a result from the dataset. It will try to answer the question using the dataset, and if it cannot, it will ask for clarification.

    Example Input: "How many rows are in table1?"
    """  # noqa: E501
    llm_chain: Any = None
    powerbi: PowerBIDataset = Field(exclude=True)
    examples: Optional[str] = DEFAULT_FEWSHOT_EXAMPLES
    session_cache: Dict[str, Any] = Field(default_factory=dict, exclude=True)
    max_iterations: int = 5
    output_token_limit: int = 4000
    tiktoken_model_name: Optional[str] = None  # "cl100k_base"

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @model_validator(mode="before")
    @classmethod
    def validate_llm_chain_input_variables(  # pylint: disable=E0213
        cls, values: dict
    ) -> dict:
        """Make sure the LLM chain has the correct input variables."""
        llm_chain = values["llm_chain"]
        for var in llm_chain.prompt.input_variables:
            if var not in ["tool_input", "tables", "schemas", "examples"]:
                raise ValueError(
                    "LLM chain for QueryPowerBITool must have input variables ['tool_input', 'tables', 'schemas', 'examples'], found %s",  # noqa: E501 # pylint: disable=C0301
                    llm_chain.prompt.input_variables,
                )
        return values

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
            logger.debug("Found cached result for %s: %s", tool_input, cache)
            return cache

        try:
            logger.info("Running PBI Query Tool with input: %s", tool_input)
            query = self.llm_chain.predict(
                tool_input=tool_input,
                tables=self.powerbi.get_table_names(),
                schemas=self.powerbi.get_schemas(),
                examples=self.examples,
                callbacks=run_manager.get_child() if run_manager else None,
            )
        except Exception as exc:  # pylint: disable=broad-except
            self.session_cache[tool_input] = f"Error on call to LLM: {exc}"
            return self.session_cache[tool_input]
        if query == "I cannot answer this":
            self.session_cache[tool_input] = query
            return self.session_cache[tool_input]
        logger.info("PBI Query:\n%s", query)
        start_time = perf_counter()
        pbi_result = self.powerbi.run(command=query)
        end_time = perf_counter()
        logger.debug("PBI Result: %s", pbi_result)
        logger.debug(f"PBI Query duration: {end_time - start_time:0.6f}")
        result, error = self._parse_output(pbi_result)
        if error is not None and "TokenExpired" in error:
            self.session_cache[tool_input] = (
                "Authentication token expired or invalid, please try reauthenticate."
            )
            return self.session_cache[tool_input]

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
            logger.debug("Found cached result for %s: %s", tool_input, cache)
            return f"{cache}, from cache, you have already asked this question."
        try:
            logger.info("Running PBI Query Tool with input: %s", tool_input)
            query = await self.llm_chain.apredict(
                tool_input=tool_input,
                tables=self.powerbi.get_table_names(),
                schemas=self.powerbi.get_schemas(),
                examples=self.examples,
                callbacks=run_manager.get_child() if run_manager else None,
            )
        except Exception as exc:  # pylint: disable=broad-except
            self.session_cache[tool_input] = f"Error on call to LLM: {exc}"
            return self.session_cache[tool_input]

        if query == "I cannot answer this":
            self.session_cache[tool_input] = query
            return self.session_cache[tool_input]
        logger.info("PBI Query: %s", query)
        start_time = perf_counter()
        pbi_result = await self.powerbi.arun(command=query)
        end_time = perf_counter()
        logger.debug("PBI Result: %s", pbi_result)
        logger.debug(f"PBI Query duration: {end_time - start_time:0.6f}")
        result, error = self._parse_output(pbi_result)
        if error is not None and ("TokenExpired" in error or "TokenError" in error):
            self.session_cache[tool_input] = (
                "Authentication token expired or invalid, please try to reauthenticate or check the scope of the credential."  # noqa: E501
            )
            return self.session_cache[tool_input]

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
    ) -> Tuple[Optional[str], Optional[Any]]:
        """Parse the output of the query to a markdown table."""
        if "results" in pbi_result:
            rows = pbi_result["results"][0]["tables"][0]["rows"]
            if len(rows) == 0:
                logger.info("0 records in result, query was valid.")
                return (
                    None,
                    "0 rows returned, this might be correct, but please validate if all filter values were correct?",  # noqa: E501
                )
            result = json_to_md(rows)
            too_long, length = self._result_too_large(result)
            if too_long:
                return (
                    f"Result too large, please try to be more specific or use the `TOPN` function. The result is {length} tokens long, the limit is {self.output_token_limit} tokens.",  # noqa: E501
                    None,
                )
            return result, None

        if "error" in pbi_result:
            if (
                "pbi.error" in pbi_result["error"]
                and "details" in pbi_result["error"]["pbi.error"]
            ):
                return None, pbi_result["error"]["pbi.error"]["details"][0]["detail"]
            return None, pbi_result["error"]
        return None, pbi_result

    def _result_too_large(self, result: str) -> Tuple[bool, int]:
        """Tokenize the output of the query."""
        if self.tiktoken_model_name:
            tiktoken_ = _import_tiktoken()
            encoding = tiktoken_.encoding_for_model(self.tiktoken_model_name)
            length = len(encoding.encode(result))
            logger.info("Result length: %s", length)
            return length > self.output_token_limit, length
        return False, 0


class InfoPowerBITool(BaseTool):
    """Tool for getting metadata about a PowerBI Dataset."""

    name: str = "schema_powerbi"
    description: str = """
    Input to this tool is a comma-separated list of tables, output is the schema and sample rows for those tables.
    Be sure that the tables actually exist by calling list_tables_powerbi first!

    Example Input: "table1, table2, table3"
    """  # noqa: E501
    powerbi: PowerBIDataset = Field(exclude=True)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

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

    name: str = "list_tables_powerbi"
    description: str = "Input is an empty string, output is a comma separated list of tables in the database."  # noqa: E501 # pylint: disable=C0301
    powerbi: PowerBIDataset = Field(exclude=True)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

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
