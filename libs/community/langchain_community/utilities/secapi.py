"""Custom SEC API wrapper for accessing various SEC EDGAR endpoints."""
import logging
from typing import Any, Dict, List, Optional

import requests
from langchain_core.utils import get_from_dict_or_env
from pydantic import BaseModel, SecretStr, model_validator


class CustomSECAPI(BaseModel):
    """Wrapper for SEC EDGAR API endpoints."""
    
    api_key: SecretStr
    base_url: str = "https://api.sec-api.io"

    @model_validator(mode='before')
    @classmethod
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key exists in environment."""
        values["api_key"] = get_from_dict_or_env(values, "api_key", "SEC_API_KEY")
        return values
    
    def full_text_search(
        self,
        search_query: str,
        form_types: Optional[List[str]] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        limit: int = 50
    ) -> Dict[str, Any]:
        """Search the full text content of SEC filings.
        
        Args:
            search_query: Text to search for in filings (e.g. "substantial doubt")
            form_types: List of form types to search (e.g. ["10-K", "8-K"])
            date_from: Start date in YYYY-MM-DD format
            date_to: End date in YYYY-MM-DD format
            limit: Maximum number of results to return (default 50)
            
        Returns:
            Dict containing search results with filing metadata and URLs
        """
        json_data = {
            "query": f'"{search_query}"',
            "formTypes": form_types,
            "startDate": date_from if date_from else None,
            "endDate": date_to if date_to else None,
            "size": limit
        }

        json_data = {k: v for k, v in json_data.items() if v is not None}

        headers = {"Authorization": self.api_key.get_secret_value()}
        
        response = requests.post(
            f"{self.base_url}/full-text-search",
            json=json_data,
            headers=headers,
            timeout=10
        )
        
        return response.json()

    def get_filings(
        self,
        ticker: str,
        form_type: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        limit: int = 50
    ) -> Dict[str, Any]:
        """Search for company filings by ticker and form type."""
        json_data = {
            "query": f"ticker:{ticker}",
            "from": "0",
            "size": str(limit),
            "sort": [{"filedAt": {"order": "desc"}}]
        }

        if form_type:
            json_data["query"] = f"{json_data['query']} AND formType:{form_type}"
        if date_from:
            json_data["query"] = (
                f"{json_data['query']} AND filedAt:[{date_from} TO {date_to or '*'}]"
            )

        headers = {
            "Authorization": self.api_key.get_secret_value(),
            "Content-Type": "application/json"
        }
        
        logging.debug("Request: %s", json_data)
        
        response = requests.post(
            "https://api.sec-api.io",
            json=json_data,
            headers=headers,
            timeout=10
        )
        
        return response.json()
    