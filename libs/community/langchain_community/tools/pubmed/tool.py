from typing import Optional

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import Field

from langchain_community.utilities.pubmed import PubMedAPIWrapper


class PubmedQueryRun(BaseTool):
    """Tool that searches the PubMed API."""

    name: str = "pub_med"
    description: str =  (
        "A wrapper around PubMed. "
        "Useful for when you need to answer questions about medicine, health, "
        "and biomedical topics from biomedical literature, MEDLINE, life science journals, and online books. "
        "Input should be a search query."
        "The query has special syntax for different fields in the paper:\n"
        "The list of available search fields is: All Fields, Author, Date - Create, Date - Publication, EC/RN Number, Editor, Title, Title/Abstract, Transliterated Title, Volume"
        "For some fields i.e. the date fields ranges are available. Here is an example of a date query and its usage.\n"
        "List papers that are from 2010: (\"2010/01/01\"[Date - Entry] : \"2011/01/01\"[Date - Entry])\n"
        "Most of the other fields are used as follows:\n"
        "List papers with first author John Doe: John Doe[Author].\n"
        "In order to combine a filter with multiple fields we use AND. Here is an example\n"
        "What are the papers with Last Author Ivan from 2020 until now:"
        "(Ivan[Author - Last]) AND ((\"2020/02\"[Date - Create] : \"3000\"[Date - Create]))")
    api_wrapper: PubMedAPIWrapper = Field(default_factory=PubMedAPIWrapper)  # type: ignore[arg-type]

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the PubMed tool."""
        return self.api_wrapper.run(query)
