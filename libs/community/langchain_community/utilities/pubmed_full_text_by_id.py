from __future__ import absolute_import, print_function, unicode_literals

from builtins import str
from typing import Optional

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import Field
from langchain_core.tools import BaseTool
from langchain_community.utilities.parsing_logic import PubMed_Central_Parser
from bs4 import BeautifulSoup


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
        """Default run is to extract full text by PMCID."""
        return self.run_pmcid(query)

    def run_pmcid(self, query: str) -> str:
        """
        Run PMC_ID(Pubmed Central ID) search and get the full text.
        Call parser to download pmc paper first, then read_and_extract_xml_data
        """

        try:
            self.api_wrapper.download_pmc_s3(query)
        except Exception as e:
            raise ("This pmcid is wrong or the pmcid doesn't have free full text")

        try:
            fileName = str("pmc/" + query + ".xml")
            text = self.read_and_extract_xml_data(fileName)
            return text
        except Exception as e:
            print(f"Error read and extract pmcid {query}: {e}")

    def run_pmid(self, query: str) -> str:
        """
        Run PMID(Pubmed ID) search and get the full text.
        Convert PMID to PMCID first, then call run_pmcid
        """
        try:
            ids = self.api_wrapper.id_lookup(query)
            pmcid = ids.get('pmcid')
            return self.run_pmcid(pmcid)

        except Exception as e:
            raise (f"This pmid is wrong: {query}")

    def read_and_extract_xml_data(self, fileName):
        with open(fileName, "r") as f:
            data = f.read()
        xml_data = BeautifulSoup(data, "xml")
        text = self.api_wrapper.extract_text(xml_data)
        return text