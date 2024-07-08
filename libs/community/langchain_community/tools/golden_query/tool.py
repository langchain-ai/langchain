"""Tool for the Golden API."""

from typing import Optional

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool

from langchain_community.utilities.golden_query import GoldenQueryAPIWrapper


class GoldenQueryRun(BaseTool):
    """Tool that adds the capability to query using the Golden API and get back JSON."""

    name: str = "golden_query"
    description: str = (
        "A wrapper around Golden Query API."
        " Useful for getting entities that match"
        " a natural language query from Golden's Knowledge Base."
        "\nExample queries:"
        "\n- companies in nanotech"
        "\n- list of cloud providers starting in 2019"
        "\nInput should be the natural language query."
        "\nOutput is a paginated list of results or an error object"
        " in JSON format."
    )
    api_wrapper: GoldenQueryAPIWrapper

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the Golden tool."""
        return self.api_wrapper.run(query)
