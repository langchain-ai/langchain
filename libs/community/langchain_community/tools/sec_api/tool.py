"""Tool for the SEC API."""
from typing import Optional, List
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_community.utilities.secapi import CustomSECAPI

class SECAPITool(BaseTool):
    """Tool that provides access to various SEC API endpoints."""
    
    name: str = "sec_api"
    description: str = """Tool for accessing SEC EDGAR data including full text search and filing search."""
    
    def __init__(self, api_key: str = "50ccec65f402053834331285c5702a5bdd3febeb66c8e25ce34b51259b5b735f", **kwargs):
        """Initialize the SEC API tool."""
        super().__init__(**kwargs)
        self._api_wrapper = CustomSECAPI(api_key=api_key)

    @property
    def api_wrapper(self) -> CustomSECAPI:
        """Get the API wrapper."""
        return self._api_wrapper

    def full_text_search(
        self,
        query: str,
        form_types: Optional[List[str]] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        limit: int = 50
    ) -> str:
        """Search the full text content of SEC filings."""
        try:
            results = self.api_wrapper.full_text_search(
                search_query=query,
                form_types=form_types,
                date_from=date_from,
                date_to=date_to,
                limit=limit
            )
            return str(results)
        except Exception as e:
            return f"Error in full text search: {str(e)}"

    def filing_search(
        self,
        ticker: str,
        form_type: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        limit: int = 50
    ) -> str:
        """Search SEC filings by company ticker."""
        try:
            results = self.api_wrapper.get_filings(
                ticker=ticker,
                form_type=form_type,
                date_from=date_from,
                date_to=date_to,
                limit=limit
            )
            return str(results)
        except Exception as e:
            return f"Error in filing search: {str(e)}"

    def _run(self, *args, **kwargs) -> str:
        """Use tool."""
        raise NotImplementedError(
            "SECAPITool cannot be run directly. Please use full_text_search() or filing_search() methods."
        )