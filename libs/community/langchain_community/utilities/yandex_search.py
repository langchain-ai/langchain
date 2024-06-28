import warnings
from typing import Any, Dict, List, Optional

import requests
from defusedxml import ElementTree as ET
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.utils import get_from_dict_or_env


class YandexSearchAPIWrapper(BaseModel):
    """Wrapper for Yandex Search API.

    Adapted from instructions available in Yandex Search API documentation.

    TODO: DOCS for using it
    - Sign up for a Yandex account if you don't already have one.
    - Create a project in the Yandex Cloud Console to use the Yandex Search API.
    - Follow the Yandex documentation to get an API key.

    Example documentation link: [https://yandex.cloud/en/docs/search-api/]
    """

    api_key: Optional[str] = None
    """The service account's API key.
    Used for user authentication together with the folder ID."""

    yandex_folder_id: Optional[str] = None
    """The folder ID of the service account used to send requests.
    Required for user authentication together with the API key."""

    k: int = 10
    """Maximum number of groups that can be returned per page with search results. 
    The range of possible values is 1 to 100."""
    search_params: Optional[Dict[str, Any]] = None
    filter: str = "moderate"  # moderate by default, can be changed to none or strict
    lr: Optional[int] = 225  # Region identifier, default is Russia
    l10n: str = "ru"  # Default notification language

    class Config:
        extra = Extra.forbid

    def _get_yandex_domain(self, language: str) -> str:
        """Determine the appropriate Yandex domain based on
        the notification language."""
        if language == "tr":
            return "https://yandex.com.tr/search/xml"
        elif language == "en":
            return "https://yandex.com/search/xml"
        else:  # Covers 'ru', 'uk', 'be', 'kk'
            return "https://yandex.ru/search/xml"

    def _yandex_search_results(
        self,
        search_term: str,
        page: int = 1,
        num_results: int = 10,
        filter: Optional[str] = None,
        lr: Optional[int] = None,
        l10n: Optional[str] = None,
    ) -> List[dict]:
        """
        Sends a POST request to the Yandex Search API and returns the search results.

        Args:
            search_term (str): The search query string.
            page (int, optional): The page number of the search results.
                Defaults to 1.
            num_results (int, optional): The number of search results per page.
                Defaults to 10.
            filter (str, optional): The filter setting ('none', 'moderate', 'strict').
                If None, the default filter setting is used.
            lr (int, optional): The region identifier for localized search.
                If None, the default region is used.
            l10n (str, optional): The localization language setting
                (e.g., 'ru', 'en', 'tr'). If None, the default language is used.

        Returns:
            List[dict]: A list of dictionaries, each containing data about a
            single search result.

        Each dictionary in the returned list contains:
            - 'url': URL of the document.
            - 'title': Title of the document.
            - 'domain': Domain name of the URL.
            - 'snippet': Text snippet from the document.
        """

        domain_url = self._get_yandex_domain(l10n if l10n is not None else self.l10n)
        headers: Dict[str, str] = {}
        params = {
            "folderid": self.yandex_folder_id,
            "apikey": self.api_key,
            "filter": filter if filter is not None else self.filter,
            "lr": lr if lr is not None else self.lr,
            "l10n": l10n if l10n is not None else self.l10n,
        }
        body = f"""<?xml version="1.0" encoding="UTF-8"?>
                    <request>
                    <query>{search_term}</query>
                    <page>{page}</page>
                    <groupings>
                        <groupby attr="d" mode="deep" 
                            groups-on-page="{num_results}" docs-in-group="1" />
                    </groupings>
                    </request>""".encode("utf-8")

        response = requests.post(domain_url, params=params, headers=headers, data=body)

        if response.status_code == 200:
            root = ET.fromstring(response.content)
            ya_respoonse = root.find("response")

            if ya_respoonse is None:
                warnings.warn("Invalid XML response format.", Warning, stacklevel=3)
                return []

            if ya_respoonse.findall("./error"):
                warnings.warn(
                    f"API Error: {ya_respoonse.findall('./error')[0].attrib}\n"
                    "Please check: "
                    "https://yandex.cloud/en/docs/search-api/reference/error-codes",
                    Warning,
                    stacklevel=3,
                )
                return []

            items = []
            for group in root.findall(".//group"):
                for doc in group.iter("doc"):
                    passages = doc.findall(".//passage")
                    passage_text = ""
                    if len(passages):
                        for passage in passages:
                            text_parts = [
                                f"{part.strip()} " for part in passage.itertext()
                            ]
                            passage_text = "".join(text_parts).strip()

                    item = {
                        "url": doc.find("url").text
                        if doc.find("url") is not None
                        else "",
                        "title": "".join(doc.find("title").itertext())
                        if doc.find("title") is not None
                        else "",
                        "domain": doc.find("domain").text
                        if doc.find("domain") is not None
                        else "",
                        "snippet": passage_text,
                    }
                    items.append(item)

            return items
        else:
            warnings.warn(
                f"HTTP Error: {response.status_code}\n"
                "Please check: https://yandex.cloud/en/docs/search-api/",
                Warning,
                stacklevel=5,
            )
            return []

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that the API key exists in the environment."""
        api_key = get_from_dict_or_env(values, "api_key", "YANDEX_API_KEY")
        folder_id = get_from_dict_or_env(values, "yandex_folder_id", "YANDEX_FOLDER_ID")
        values["api_key"] = api_key
        values["yandex_folder_id"] = folder_id

        return values

    def run(
        self, 
        query: str, 
        filter: Optional[str] = None, 
        lr: Optional[int] = None, 
        l10n: Optional[str] = None
    ) -> str:
        """
        Run query through Yandex Search and parse result.

        Args:
            query (str): The search query string.
            filter (str, optional): The filter setting ('none', 'moderate', 'strict').
            lr (int, optional): The region identifier for localized search.
            l10n (str, optional): The language for notifications
                (e.g., 'ru', 'en', 'tr').

        Returns:
            str: Concatenated string of snippets if any results are found,
            otherwise a not found message.
        """

        snippets = []
        results = self._yandex_search_results(
            search_term=query, num_results=self.k, filter=filter, lr=lr, l10n=l10n
        )
        if len(results) == 0:
            return "No good Yandex Search Result was found"

        snippets = [result["snippet"] for result in results if "snippet" in result]
        return " ".join(snippets)

    def results(
        self,
        query: str,
        num_results: int,
        search_params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict]:
        """Run query through Yandex Search and return metadata.

        Args:
            query: The query to search for.
            num_results: The number of results to return.
            search_params: Parameters to be passed on search

        Returns:
            A list of dictionaries with keys depending on
            the Yandex Search API response structure.
        """
        metadata_results = []
        results = self._yandex_search_results(
            query, num_results=num_results, **(search_params or {})
        )
        if len(results) == 0:
            return [{"Result": "No good Yandex Search Result was found"}]
        for result in results:
            metadata_result = {
                "title": result.get("title"),
                "link": result.get("url"),
                "snippet": result.get("snippet"),
            }
            metadata_results.append(metadata_result)

        return metadata_results
