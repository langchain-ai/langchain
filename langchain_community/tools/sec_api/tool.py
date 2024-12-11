"""Tool for querying SEC filings."""
from typing import Optional, Any, Dict
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_community.utilities.secapi import CustomSECQueryAPI

class SECFilingInput(BaseModel):
    """Input for SEC filing search."""
    ticker: str = Field(..., description="Company ticker symbol")
    form_type: Optional[str] = Field(None, description="SEC form type (e.g., '10-K', '10-Q')")
    date_from: Optional[str] = Field(None, description="Start date in YYYY-MM-DD format")
    date_to: Optional[str] = Field(None, description="End date in YYYY-MM-DD format")
    limit: Optional[int] = Field(50, description="Maximum number of results to return")

class SECFilingsTool(BaseModel):
    """Tool that searches SEC EDGAR filings."""
    
    name: str = "sec_filings"
    description: str = """
    Search for SEC filings by company ticker and optional parameters.
    Useful for finding financial reports and company disclosures.
    """
    api_key: str
    api: Optional[CustomSECQueryAPI] = None
    
    def __init__(self, **data: Any):
        super().__init__(**data)
        self.api = CustomSECQueryAPI(api_key=self.api_key)
    
    def _run(
        self,
        ticker: str,
        form_type: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        limit: Optional[int] = 50
    ) -> str:
        """Run SEC filing search."""
        try:
            results = self.api.get_filings(
                ticker=ticker,
                form_type=form_type,
                date_from=date_from,
                date_to=date_to,
                limit=limit
            )
            return str(results)
        except Exception as e:
            return f"Error searching SEC filings: {str(e)}"

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        """Run SEC filing search asynchronously."""
        return self._run(*args, **kwargs)

    def run(self, query: Dict[str, Any]) -> str:
        """Run the tool with the given query."""
        return self._run(**query)