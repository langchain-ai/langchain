from __future__ import absolute_import, print_function, unicode_literals
from typing import Optional
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import Field
from langchain_core.tools import BaseTool
from langchain_community.utilities import PmcIDSearchWrapper 
from builtins import str


class PmcIDyRun(BaseTool):
    """Tool that searches the PubMed API."""

    name: str = "pub_med"
    description: str = (
        "A wrapper around PubMed Central. "
        "Returned the full-text of PMC paper "
        "Input should be a PMCID."
    )
    api_wrapper: PmcIDSearchWrapper = Field(default_factory=PmcIDSearchWrapper)  # type: ignore[arg-type]

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the PubMed tool."""
        return self.api_wrapper.run(query)
