from builtins import str
from typing import Optional
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import Field
from langchain_core.tools import BaseTool
from parsing_logic import PubMed_Central_Parser


class PmcIDyRun(BaseTool):
    """Tool that searches the PubMed API."""

    name: str = "pub_med"
    description: str = (
        "A wrapper around PubMed Central. "
        "Returned the full-text of PMC paper "
        "Input should be a PMCID."
    )
    api_wrapper: PubMed_Central_Parser = Field(default_factory=PubMed_Central_Parser)  # type: ignore[arg-type]

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """
        Run PMC_ID(Pubmed Central ID) search and get the full text.
        Call parser to get xml of pmc paper first, then extract text from xml
        """
        xml_string = self.api_wrapper.get_xml(query)
        if xml_string is None:
            return f"Error: Unable to retrieve XML data for PMC ID. The XML of the {query} might not be available."
        text = self.api_wrapper.extract_text(xml_string)
        return text
