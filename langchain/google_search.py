"""Chain that calls Google Search.

"""
import os
import sys
from typing import Any, Dict, Optional

from pydantic import BaseModel, Extra, root_validator

from langchain.utils import get_from_dict_or_env
from googleapiclient.discovery import build

class HiddenPrints:
    """Context manager to hide prints."""

    def __enter__(self) -> None:
        """Open file to pipe stdout to."""
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, *_: Any) -> None:
        """Close file that stdout was piped to."""
        sys.stdout.close()
        sys.stdout = self._original_stdout



class GoogleSearchAPIWrapper(BaseModel):
    """
    TODO: DOCS for using it

    google-api-python-client==2.70.0

    """

    search_engine: Any  #: :meta private:

    google_api_key: Optional[str] = None
    google_cse_id: Optional[str] = None

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
    
    def _google_search_results(search_term, api_key, cse_id, **kwargs):
        service = build("customsearch", "v1", developerKey=api_key)
        res = service.cse().list(q=search_term, cx=cse_id, **kwargs).execute()
        print(res)
        return res['items']

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        google_api_key = get_from_dict_or_env(
            values, "google_api_key", "GOOGLE_API_KEY"
        )
        values["google_api_key"] = google_api_key

        google_cse_id = get_from_dict_or_env(
            values, "google_cse_id", "GOOGLE_CSE_ID"
        )
        values[google_cse_id] = google_cse_id

        try:    
            from googleapiclient.discovery import build
        except ImportError:
            raise ImportError(
                "google-api-python-client is not installed. "
                "Please install it with pip install google-api-python-client"
            )

        # TODO: Add error handling if package is not installed
        # TODO: Add error handling if keys are missing
        return values

    def run(self, query: str) -> str:
        """Run query through GoogleSearch and parse result."""
        
        try:
            snippets = []
            print("Querying Google Search API... ")
            print(self.google_api_key)
            print(self.google_cse_id)
            print(query)
            results = self._google_search_results(query, self.google_api_key, self.google_cse_id, num=3)
            print(results)
            for result in results:
                snippets.append(result["snippet"])
            toret = " ".join(snippets)
        except Exception as e:
            raise ValueError("Error in Google Search API, make sure you have GOOGLE_API_KEY and GOOGLE_CSE_ID set on your enviroment.")
        

        else:
            toret = "No good search result found"
        return toret


# For backwards compatability

GoogleSearchAPIChain = GoogleSearchAPIWrapper