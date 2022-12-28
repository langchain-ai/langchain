"""Chain that calls Google Search."""
import os
import sys
from typing import Any, Dict, List, Optional

from googleapiclient.discovery import build
from pydantic import BaseModel, Extra, root_validator

from langchain.utils import get_from_dict_or_env


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
    """Wrapper for Google Search API.

    TODO: DOCS for using it
    1. Install google-api-python-client
    - If you don't already have a Google account, sign up.
    - If you have never created a Google APIs Console project,
    read the Managing Projects page and create a project in the Google API Console.
    - Install the library using pip install google-api-python-client
    The current version of the library is 2.70.0 at this time

    2. To create an API key:
    - Navigate to the APIs & Services→Credentials panel in Cloud Console.
    - Select Create credentials, then select API key from the drop-down menu.
    - The API key created dialog box displays your newly created key.
    - You now have an API_KEY

    3. Setup Custom Search Engine so you can search the entire web
    - Create a custom search engine in this link.
    - In Sites to search, add any valid URL (i.e. www.stackoverflow.com).
    - That’s all you have to fill up, the rest doesn’t matter.
    In the left-side menu, click Edit search engine → {your search engine name}
    → Setup Set Search the entire web to ON. Remove the URL you added from
     the list of Sites to search.
    - Under Search engine ID you’ll find the search-engine-ID.

    4. Enable the Custom Search API
    - Navigate to the APIs & Services→Dashboard panel in Cloud Console.
    - Click Enable APIs and Services.
    - Search for Custom Search API and click on it.
    - Click Enable.
    URL for it: https://console.cloud.google.com/apis/library/customsearch.googleapis
    .com
    Adapted from: Instructions adapated from https://stackoverflow.com/questions/
    37083058/
    programmatically-searching-google-in-python-using-custom-search


    """

    search_engine: Any  #: :meta private:

    google_api_key: Optional[str] = None
    google_cse_id: Optional[str] = None

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    def _google_search_results(
        self,
        search_term: str,
        api_key: Optional[str],
        cse_id: Optional[str],
        **kwargs: Any
    ) -> List[dict]:
        service = build("customsearch", "v1", developerKey=api_key)
        res = service.cse().list(q=search_term, cx=cse_id, **kwargs).execute()
        return res["items"]

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        google_api_key = get_from_dict_or_env(
            values, "google_api_key", "GOOGLE_API_KEY"
        )
        values["google_api_key"] = google_api_key

        google_cse_id = get_from_dict_or_env(values, "google_cse_id", "GOOGLE_CSE_ID")
        values[google_cse_id] = google_cse_id

        try:
            from googleapiclient.discovery import build

            build("customsearch", "v1", developerKey=google_api_key)
        except ImportError:
            raise ImportError(
                "google-api-python-client is not installed. "
                "Please install it with pip install google-api-python-client"
            )

        # TODO: Add error handling if keys are missing
        return values

    def run(self, query: str) -> str:
        """Run query through GoogleSearch and parse result."""
        try:
            snippets = []
            results = self._google_search_results(
                query, self.google_api_key, self.google_cse_id, num=10
            )
            if len(results) == 0:
                return "No good Google Search Result was found"
            for result in results:
                snippets.append(result["snippet"])

            return " ".join(snippets)

        except Exception:
            raise ValueError(
                """Error in Google Search API, make sure you have
                GOOGLE_API_KEY and GOOGLE_CSE_ID set on your enviroment.
                If you have exceeded your 10,000/day token, please refer
                to the google cse documentation."""
            )


# For backwards compatability
GoogleSearchAPIChain = GoogleSearchAPIWrapper
