"""Chain that calls Oxylabs API for Google Search.

"""

import os
import sys
from typing import Any, Dict, Optional, Tuple, List

import asyncio
from langchain_core.utils import get_from_dict_or_env
from pydantic import BaseModel, ConfigDict, Field, model_validator


def _get_default_params() -> dict:
    """ Provide default parameters for OxylabsSearchAPIWrapper.

    Returns:
        dict: Default parameters, including the following keys:
            - oxylabs_username (str): Oxylabs username, either provided directly or via the environment variable `OXYLABS_USERNAME`.
            - oxylabs_password (str): Oxylabs password, either provided directly or via the environment variable `OXYLABS_PASSWORD`.
            - engine (str): Search engine to be used, e.g., "google".
            - source (str): Source of the search engine, e.g., "google_search".
            - user_agent_type (str): User agent type, e.g., "desktop". Can be set using values from `oxylabs.utils.types`.
            - render (str): Render type for the search results, e.g., "html". Can be set using values from `oxylabs.utils.types`.
            - domain (str): Domain for the search engine, e.g., "com".
            - start_page (int): Starting page number for search results.
            - pages (int): Number of pages to retrieve.
            - limit (int): Maximum number of results to return.
            - parse (bool): Whether to enable result parsing into an object.
            - locale (str): Locale or location for the search.
            - geo_location (str): Geographic location for the search.
            - callback_url (str): URL for callbacks (if applicable).
            - parsing_instructions (dict): Additional instructions for parsing the search results.
            - context (list): Search context information.
            - request_timeout (int): Timeout for the Oxylabs service, in seconds.
    """

    return {
        "oxylabs_username": "",
        "oxylabs_password": "",
        "engine": "google",
        "source": "google_search",
        "user_agent_type": "desktop",
        "render": "html",
        "domain": "com",
        "start_page": 1,
        "pages": 3,
        "limit": 5,
        "parse": True,
        "locale": "",
        "geo_location": "",
        "callback_url": "",
        "parsing_instructions": {},
        "context": [],
        "request_timeout": 165,
    }


class OxylabsSearchAPIWrapper(BaseModel):
    """ Wrapper class for OxylabsSearchAPI.

    Example:
        ```python
        from langchain_community.utilities import OxylabsSearchAPIWrapper

        oxylabs_api = OxylabsSearchAPIWrapper(
            params={
                "oxylabs_username": <OXYLABS_USERNAME>,
                "oxylabs_password": <OXYLABS_PASSWORD>,
                "engine": "google",
                "source": "google_search",
                "user_agent_type": "desktop",
                "render": "html",
                "domain": "com",
                "start_page": 1,
                "pages": 1,
                "limit": 5,
                "parse": True,
                "locale": "",
                "geo_location": "",
                "callback_url": "",
                "parsing_instructions": {},
                "context": [],
                "request_timeout": 165,
            }
        )
        search_query = "Oxylabs"
        result = oxylabs_api.run(search_query)
        print(result)
        ```
    """

    search_engine: Any = None  #: :meta private:
    params: dict = Field(
        default=_get_default_params
    )
    oxylabs_username: Optional[str] = None
    oxylabs_password: Optional[str] = None

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra='forbid'
    )

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """Validate that oxylabs username, password and python package exists in environment."""

        default_params = _get_default_params()
        default_params.update(**values.get("params", {}))
        current_params = default_params
        oxylabs_username = get_from_dict_or_env(
            current_params, "oxylabs_username", "OXYLABS_USERNAME"
        )
        oxylabs_password = get_from_dict_or_env(
            current_params, "oxylabs_password", "OXYLABS_PASSWORD"
        )
        if not (oxylabs_username and oxylabs_password):
            raise RuntimeError(
                "Please set up environment variables: `OXYLABS_USERNAME` and `OXYLABS_PASSWORD`"
                " or provide credentials in the parameters as `oxylabs_username` and `oxylabs_password`."
            )

        formed_values = dict()
        formed_values["oxylabs_username"] = oxylabs_username
        formed_values["oxylabs_password"] = oxylabs_password
        formed_values["params"] = dict(current_params)

        if current_params["engine"] not in [
            "google",
            # "bing",
            # "amazon",
            # "universal"
        ]:
            raise NotImplementedError(
                f"The search engine {current_params['engine']} is not supported at the moment."
            )

        try:
            from oxylabs import RealtimeClient

        except ImportError:
            raise ImportError(
                "Could not import oxylabs python package. "
                "Please install it with `pip install oxylabs`."
            )

        try:
            oxylabs_realtime_client = RealtimeClient(oxylabs_username, oxylabs_password)
            # The process to set any available provider
            oxylabs_realtime_client_by_provider = getattr(
                oxylabs_realtime_client.serp,
                current_params["engine"]
            )
            formed_values["search_engine"] = oxylabs_realtime_client_by_provider

        except Exception as exc:
            raise RuntimeError(f"Unknown Oxylabs Python SDK integration error: {exc}")

        return formed_values

    async def arun(self, query: str, **kwargs: Any) -> str:
        """Run query through OxylabsSearchAPI and parse result async."""
        return self._process_response(await self.aresults(query))

    def run(self, query: str, **kwargs: Any) -> str:
        """Run query through OxylabsSearchAPI and parse result."""
        return self._process_response(self.results(query))

    def results(self, query: str) -> List[Dict[str, Any]]:
        """Run query through Oxylabs Web Scrapper API and return SERPResponse object."""
        params_ = self.get_params()
        search_client = self.search_engine
        search_result = search_client.scrape_search(query, **params_)

        return search_result

    async def aresults(self, query: str) -> List[Dict[str, Any]]:
        """Run query through Oxylabs Web Scrapper API and return SERPResponse object async."""
        params_ = self.get_params()

        search_client = self.search_engine
        search_result = await asyncio.to_thread(
            search_client.scrape_search,
            query,
            **params_,
        )

        return search_result

    def get_params(self) -> Dict[str, Any]:
        """Get default configuration parameters for OxylabsSearchAPI for scrape_search()."""
        _param_keys = list(self.params.keys())
        setup_keys = ["oxylabs_username", "oxylabs_password", "engine"]
        for setup_key in setup_keys:
            _param_keys.remove(setup_key)

        _params = {
            f"{p_key}": self.params[p_key] for p_key in _param_keys
            if self.params[p_key]
        }

        return _params

    def _validate_response(self, response: Any) -> dict:
        """Validate Oxylabs SERPResponse format and unpack data."""
        try:
            result_list = response.raw['results']
            if not isinstance(result_list, list) or not result_list:
                raise ValueError("No results returned!")

            results = dict(result_list[0])
            content = results['content']
            if not isinstance(content, dict):
                raise ValueError(
                    "Result `content` format error, try setting parameter `parse` to True"
                )

            unpacked_results = content['results']
            if not isinstance(unpacked_results, dict):
                raise ValueError("Response format Error!")

            return unpacked_results

        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Response Validation Error: {str(exc)}")

    def _process_response(self, res: Any) -> str:
        """Process Oxylabs SERPResponse and serialize search results to string."""

        result_ = "No good search result found"

        try:
            validated_response = self._validate_response(res)

        except RuntimeError as exc:
            return f"{result_}, Response Validation Failed: {str(exc)}"

        snippets = list()

        # Knowledge Graph Snippets
        self._create_knowledge_graph_snippets(validated_response, snippets)

        # Combined Search Result Snippets [Organic Results, Paid Results, Popular Products, Related Searches]
        self._create_combined_search_result_snippets(validated_response, snippets)

        # Search Information Snippets
        self._create_search_information_snippets(validated_response, snippets)

        # Combine all snippets
        if snippets:
            result_ = "\n\n".join(snippets)

        return result_

    def _create_search_information_snippets(
            self,
            results: dict,
            search_information_snippets: list
    ) -> None:
        """Create search information snippets from Oxylabs SERPResponse data search_information tag."""

        if (
            'search_information' in results
            and isinstance(results['search_information'], dict)
            and results['search_information']
        ):
            search_information_snippets.append(f"Search Information: ")
            search_info = results['search_information']
            query = search_info.get('query', '')
            total_results_count = search_info.get('total_results_count', '')
            if query:
                search_information_snippets.append(f"Search Query: {query}")
            if total_results_count:
                search_information_snippets.append(f"Total Results: {total_results_count}")

    def _create_knowledge_graph_snippets(
            self,
            results: dict,
            knowledge_graph_snippets: list
    ) -> None:
        """Create knowledge graph snippets from Oxylabs SERPResponse data knowledge tag."""

        if (
            'knowledge' in results
            and isinstance(results['knowledge'], dict)
            and results['knowledge']
        ):
            knowledge = results['knowledge']
            self.process_knowlege_metadata(knowledge, knowledge_graph_snippets)
            self.process_knowledge_factoids(knowledge, knowledge_graph_snippets)
            self.process_knowledge_profiles(knowledge, knowledge_graph_snippets)

    def process_knowledge_profiles(
            self,
            knowledge: dict,
            knowledge_graph_snippets: list
    ) -> None:
        """Process profiles from Oxylabs SERPResponse data knowledge tag."""
        if (
            'profiles' in knowledge
            and isinstance(knowledge['profiles'], list)
            and knowledge['profiles']
        ):
            knowledge_graph_snippets.append(f"Profiles: ")
            profiles = knowledge['profiles']
            for profile in profiles:
                profile_title = profile.get('title', '')
                profile_url = profile.get('url', '')
                if profile_title and profile_url:
                    knowledge_graph_snippets.append(
                        f"(Knowledge Profile) - {profile_title}: {profile_url}"
                    )

    def process_knowledge_factoids(
            self,
            knowledge: dict,
            knowledge_graph_snippets: list
    ) -> None:
        """Process factoids from Oxylabs SERPResponse data knowledge tag."""

        if (
            'factoids' in knowledge
            and isinstance(knowledge['factoids'], list)
            and knowledge['factoids']
        ):
            knowledge_graph_snippets.append(f"Facts: ")
            factoids = knowledge['factoids']
            for factoid in factoids:
                factoid_title = factoid.get('title', '')
                factoid_content = factoid.get('content', '')
                if factoid_title and factoid_content:
                    knowledge_graph_snippets.append(
                        f"(Fact) - {factoid_title}: {factoid_content}"
                    )
                if 'links' in factoid:
                    links = factoid['links']
                    for link in links:
                        link_title = link.get('title', '')
                        link_href = link.get('href', '')
                        if link_title and link_href:
                            knowledge_graph_snippets.append(
                                f"(Fact Link) - {factoid_title}: {link_title} ({link_href})"
                            )

    def process_knowlege_metadata(
            self,
            knowledge: dict,
            knowledge_graph_snippets: list
    ) -> None:
        """Process metadata from Oxylabs SERPResponse data knowledge tag."""

        title = knowledge.get('title', '')
        subtitle = knowledge.get('subtitle', '')
        knowledge_graph_snippets.append(f"Knowledge graph: {title}")
        knowledge_graph_snippets.append(f"Subtitle: {subtitle}")
        if (
                'description' in knowledge
                and isinstance(knowledge['description'], str)
                and knowledge['description']
        ):
            if str(knowledge['description']).startswith("Description"):
                knowledge_description = ''.join(str(knowledge['description']).split("Description")[1:])
            else:
                knowledge_description = str(knowledge['description'])

            knowledge_graph_snippets.append(f"Description: {knowledge_description}")

    def _create_combined_search_result_snippets(
            self,
            results: dict,
            search_result_snippets: list
    ) -> None:
        """Create combined search result snippets from Oxylabs SERPResponse data."""

        if (
                (
                        'organic' in results
                        and results['organic']
                        and isinstance(results['organic'], list)
                ) or (
                'paid' in results
                and results['paid']
                and isinstance(results['paid'], list)
        ) or (
                'popular_products' in results
                and results['popular_products']
                and isinstance(results['popular_products'], list)
        ) or (
                'related_searches' in results
                and results['related_searches']
                and isinstance(results['related_searches'], list)
        )
        ):
            search_result_snippets.append(f"Combined search results: ")

            self.process_organic_results(results, search_result_snippets)

            self.process_paid_results(results, search_result_snippets)

            self.process_popular_poducts(results, search_result_snippets)

            self.process_related_searches(results, search_result_snippets)

    def process_related_searches(
            self,
            results: dict,
            search_result_snippets: list
    ) -> None:
        """Process related searches from Oxylabs SERPResponse data related_searches tag."""
        if (
                'related_searches' in results
                and isinstance(results['related_searches'], list)
                and results['related_searches']
        ):
            search_result_snippets.append(f"Related Searches: ")
            related_searches_data = results['related_searches']
            related_searches_list = related_searches_data.get('related_searches', [])
            for search in related_searches_list:
                search_result_snippets.append(f"Related Search: {search}")

    def process_popular_poducts(
            self,
            results: dict,
            search_result_snippets: list
    ) -> None:
        """Process popular products from Oxylabs SERPResponse data popular_products tag."""
        if (
                'popular_products' in results
                and isinstance(results['popular_products'], list)
                and results['popular_products']
        ):
            search_result_snippets.append(f"Popular Products: ")
            popular_products = results['popular_products']
            for product in popular_products:
                items = product.get('items', [])
                for item in items:
                    snippet_parts = list()
                    title = item.get('title', '')
                    price = item.get('price', '')
                    rating = item.get('rating', '')
                    seller = item.get('seller', '')
                    if title:
                        snippet_parts.append(f"Product: {title}")
                    if price:
                        snippet_parts.append(f"Price: {price}")
                    if rating:
                        snippet_parts.append(f"Rating: {rating}")
                    if seller:
                        snippet_parts.append(f"Seller: {seller}")
                    snippet = "\n".join(snippet_parts)
                    search_result_snippets.append(snippet)

    def process_paid_results(
            self,
            results: dict,
            search_result_snippets: list
    ) -> None:
        """Process paid results from Oxylabs SERPResponse data paid tag."""

        if (
                'paid' in results
                and isinstance(results['paid'], list)
                and results['paid']
        ):
            search_result_snippets.append(f"Paid Results: ")
            paid_results = results['paid']
            for paid_result in paid_results:
                snippet_parts = list()
                title = paid_result.get('title', '')
                desc = paid_result.get('desc', '')
                url = paid_result.get('url', '')
                if title:
                    snippet_parts.append(f"Paid Ad - Title: {title}")
                if desc:
                    snippet_parts.append(f"Description: {desc}")
                if url:
                    snippet_parts.append(f"URL: {url}")
                snippet = "\n".join(snippet_parts)
                search_result_snippets.append(snippet)

    def process_organic_results(
            self,
            results: dict,
            search_result_snippets: list
    ) -> None:
        """Process organic results from Oxylabs SERPResponse data organic tag."""

        if (
                'organic' in results
                and isinstance(results['organic'], list)
                and results['organic']
        ):
            search_result_snippets.append(f"Organic Results: ")
            organic_results = results['organic']
            for organic_result in organic_results:
                snippet_parts = list()
                pos = organic_result.get('pos', '')
                title = organic_result.get('title', '')
                desc = organic_result.get('desc', '')
                url = organic_result.get('url', '')
                if pos:
                    snippet_parts.append(f"Position: {pos}")
                if title:
                    snippet_parts.append(f"Title: {title}")
                if desc:
                    snippet_parts.append(f"Description: {desc}")
                if url:
                    snippet_parts.append(f"URL: {url}")
                snippet = "\n".join(snippet_parts)
                search_result_snippets.append(snippet)
