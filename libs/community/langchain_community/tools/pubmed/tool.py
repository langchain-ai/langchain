from typing import Optional

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import Field
from langchain_core.tools import BaseTool

from langchain_community.utilities.pubmed import PubMedAPIWrapper


class PubmedQueryRun(BaseTool):
    """Tool that searches the PubMed API."""

    name: str = "pub_med"
    description: str = (
        "A wrapper around PubMed. "
        "Useful for when you need to answer questions about medicine, health, "
        "and biomedical topics "
        "from biomedical literature, MEDLINE, life science journals, and online books. "
        "Input should be a search query."
    )
    api_wrapper: PubMedAPIWrapper = Field(default_factory=PubMedAPIWrapper)

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the PubMed tool."""
        return self.api_wrapper.run(query)
