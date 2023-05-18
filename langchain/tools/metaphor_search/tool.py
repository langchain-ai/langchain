"""Tool for the Metaphor search API."""

from typing import Dict, List, Optional, Union

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.tools.base import BaseTool
from langchain.utilities.metaphor_search import MetaphorSearchAPIWrapper


class MetaphorSearchResults(BaseTool):
    """Tool that has capability to query the Metaphor Search API and get back json."""

    name = "Metaphor Search Results JSON"
    description = (
        "A wrapper around Metaphor Search. "
        "Input should be a Metaphor-optimized query. "
        "Output is a JSON array of the query results"
    )
    api_wrapper: MetaphorSearchAPIWrapper

    def _run(
        self,
        query: str,
        num_results: int,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Union[List[Dict], str]:
        """Use the tool."""
        try:
            return self.api_wrapper.results(query, num_results)
        except Exception as e:
            return repr(e)

    async def _arun(
        self,
        query: str,
        num_results: int,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Union[List[Dict], str]:
        """Use the tool asynchronously."""
        try:
            return await self.api_wrapper.results_async(query, num_results)
        except Exception as e:
            return repr(e)
