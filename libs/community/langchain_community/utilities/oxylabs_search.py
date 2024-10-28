"""Chain that calls Oxylabs API for Google Search.

"""

import os
import sys
from typing import Any, Dict, Optional, Tuple, List

import asyncio
from langchain_core.utils import get_from_dict_or_env
from pydantic import BaseModel, ConfigDict, Field, model_validator
from dataclasses import dataclass
from typing import Optional


@dataclass
class ResponseElement:
    tag: str
    display_tag: str
    path_: str
    python_type: str
    parent: Optional["ResponseElement"]


def _get_default_params() -> dict:
    """Provide default parameters for OxylabsSearchAPIWrapper.

    Returns:
        dict: Default parameters, including the following keys:
            - oxylabs_username (str): Oxylabs username, either provided directly or via the environment variable `OXYLABS_USERNAME`.
            - oxylabs_password (str): Oxylabs password, either provided directly or via the environment variable `OXYLABS_PASSWORD`.
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
        "parsing_instructions": {},
        "context": [],
        "request_timeout": 165,
    }


class OxylabsSearchAPIWrapper(BaseModel):
    """Wrapper class for OxylabsSearchAPI.

    Example:
        ```python
        from langchain_community.utilities import OxylabsSearchAPIWrapper

        oxylabs_api = OxylabsSearchAPIWrapper(
            params={
                "oxylabs_username": <OXYLABS_USERNAME>,
                "oxylabs_password": <OXYLABS_PASSWORD>,
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

    search_engine: Any = None
    params: dict = Field(default=_get_default_params)
    oxylabs_username: Optional[str] = None
    oxylabs_password: Optional[str] = None

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

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
                oxylabs_realtime_client.serp, current_params["engine"]
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
        validated_responses = self._validate_response(search_result)

        return validated_responses

    async def aresults(self, query: str) -> List[Dict[str, Any]]:
        """Run query through Oxylabs Web Scrapper API and return SERPResponse object async."""
        params_ = self.get_params()

        search_client = self.search_engine
        # TODO check async implementation (self.aiohttpsession in other wrappers)
        search_result = await asyncio.to_thread(
            search_client.scrape_search,
            query,
            **params_,
        )

        validated_responses = self._validate_response(search_result)

        return validated_responses

    def get_params(self) -> Dict[str, Any]:
        """Get default configuration parameters for OxylabsSearchAPI for scrape_search()."""
        _param_keys = list(self.params.keys())
        setup_keys = ["oxylabs_username", "oxylabs_password", "engine"]
        for setup_key in setup_keys:
            _param_keys.remove(setup_key)

        _params = {
            f"{p_key}": self.params[p_key]
            for p_key in _param_keys
            if self.params[p_key]
        }

        return _params

    def _validate_response(self, response: Any) -> List[Dict[Any, Any]]:
        """Validate Oxylabs SERPResponse format and unpack data."""
        validated_results = list()
        try:
            result_list = response.raw["results"]
            if not isinstance(result_list, list) or not result_list:
                raise ValueError("No results returned!")

            # TODO make sure this works for multi page responses <- pagination
            for result_item in result_list:
                result_item = dict(result_item)
                content = result_item["content"]
                if not isinstance(content, dict):
                    raise ValueError(
                        "Result `content` format error, try setting parameter `parse` to True"
                    )

                unpacked_results = content["results"]

                if not isinstance(unpacked_results, dict):
                    raise ValueError("Response format Error!")

                if content["results"]:
                    validated_results.append(unpacked_results)

            return validated_results

        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Response Validation Error: {str(exc)}")

    def _process_response(self, res: Any) -> str:
        """Process Oxylabs SERPResponse and serialize search results to string."""

        result_ = "No good search result found"

        snippets = list()
        # TODO update here fter pagination questions are answered
        for validated_response in res[:1]:
            # Knowledge Graph Snippets
            self._create_knowledge_graph_snippets(validated_response, snippets)

            # Combined Search Result Snippets [Organic Results, Paid Results]
            self._create_combined_search_result_snippets(validated_response, snippets)

            # Product information Snippets
            self._create_product_information_snippets(validated_response, snippets)

            # Local Group Information Snippets
            self._create_local_information_snippets(validated_response, snippets)

            # Search Information Snippets
            self._create_search_information_snippets(validated_response, snippets)

        # Combine all snippets
        if snippets:
            result_ = "\n\n".join(snippets)

        return result_

    def recursive_snippet_collector(
        self,
        target_structure: Any,
        max_depth: int,
        current_depth: int,
        parent_: Optional[ResponseElement] = None,
    ) -> str:
        target_snippets = list()

        recursion_padding = "  " * (current_depth + 1)

        base64_image_attributes = ["image_data".upper(), "data".upper()]
        base64_images_attribute = "images".upper()

        if current_depth >= max_depth:
            return "\n".join(target_snippets)

        if isinstance(target_structure, (str, float, int)):
            if target_structure:
                if parent_.python_type == str(type(list())):
                    if (
                        base64_images_attribute in parent_.path_.split("-")[-3:]
                        or parent_.tag in base64_image_attributes
                    ):
                        target_structure = "Redacted base64 image string..."

                    target_snippets.append(
                        f"{recursion_padding}{parent_.display_tag}: {str(target_structure)}"
                    )

                elif parent_.python_type == str(type(dict())):
                    if parent_.tag in base64_image_attributes:
                        target_structure = "Redacted base64 image string..."

                    target_snippets.append(
                        f"{recursion_padding}{parent_.display_tag}: {str(target_structure)}"
                    )

        elif isinstance(target_structure, dict):
            if target_structure:
                target_snippets.append(
                    f"{recursion_padding}{parent_.display_tag.upper()}: "
                )
                for key_, value_ in target_structure.items():
                    if isinstance(value_, dict):
                        if value_:
                            target_snippets.append(
                                f"{recursion_padding}{key_.upper()}: "
                            )
                            target_snippets.append(
                                self.recursive_snippet_collector(
                                    value_,
                                    max_depth=max_depth,
                                    current_depth=current_depth + 1,
                                    parent_=ResponseElement(
                                        path_=f"{parent_.path_.upper()}-{key_.upper()}",
                                        tag=key_.upper(),
                                        display_tag=key_.upper(),
                                        python_type=str(type(value_)),
                                        parent=parent_,
                                    ),
                                )
                            )

                    elif isinstance(value_, (list, tuple)):
                        if value_:
                            target_snippets.append(
                                f"{recursion_padding}{key_.upper()} ITEMS: "
                            )
                            for nr_, _element in enumerate(value_):
                                target_snippets.append(
                                    self.recursive_snippet_collector(
                                        _element,
                                        max_depth=max_depth,
                                        current_depth=current_depth + 1,
                                        parent_=ResponseElement(
                                            path_=f"{parent_.path_.upper()}-{key_.upper()}-ITEM-{nr_ + 1}",
                                            tag=key_.upper(),
                                            display_tag=f"{key_.upper()}-ITEM-{nr_ + 1}",
                                            python_type=str(type(value_)),
                                            parent=parent_,
                                        ),
                                    )
                                )

                    elif isinstance(value_, (str, float, int)):
                        if value_:
                            if key_.upper() in base64_image_attributes:
                                value_ = "Redacted base64 image string..."

                            target_snippets.append(
                                f"{recursion_padding}{key_.upper()}: {str(value_)}"
                            )

        elif isinstance(target_structure, (list, tuple)):
            if target_structure:
                target_snippets.append(
                    f"{recursion_padding}{parent_.display_tag.upper()} ITEMS: "
                )
            for nr_, element_ in enumerate(target_structure):
                target_snippets.append(
                    self.recursive_snippet_collector(
                        element_,
                        max_depth=max_depth,
                        current_depth=current_depth + 1,
                        parent_=ResponseElement(
                            path_=f"{parent_.path_.upper()}-ITEM-{nr_ + 1}",
                            tag=parent_.tag.upper(),
                            display_tag=f"{parent_.tag.upper()}-ITEM-{nr_ + 1}",
                            python_type=str(type(target_structure)),
                            parent=parent_,
                        ),
                    )
                )

        return "\n".join(target_snippets)

    def process_tags(self, snippets_, tags_, results, group_name: str = ""):
        check_tags = [tag_[0] in results for tag_ in tags_]
        if any(check_tags):
            for tag in tags_:
                tag_content = results.get(tag[0], {}) or {}
                if tag_content:
                    collected_snippets = self.recursive_snippet_collector(
                        tag_content,
                        max_depth=5,
                        current_depth=0,
                        parent_=ResponseElement(
                            path_=f"{group_name}-{tag[0]}",
                            tag=tag[0],
                            display_tag=tag[1],
                            python_type=str((type(tag_content))),
                            parent=None,
                        ),
                    )
                    if collected_snippets:
                        snippets_.append(collected_snippets)

    def _create_knowledge_graph_snippets(
        self, results: dict, knowledge_graph_snippets: list
    ) -> None:
        """Create knowledge graph snippets from Oxylabs SERPResponse data search_information tag."""

        knowledge_graph_tags = [
            ("knowledge", "Knowledge Graph"),
            ("recipes", "Recipes"),
            ("item_carousel", "Item Carousel"),
            ("apps", "Apps"),
        ]
        self.process_tags(
            knowledge_graph_snippets, knowledge_graph_tags, results, "Knowledge"
        )

    def _create_combined_search_result_snippets(
        self, results: dict, combined_search_result_snippets: list
    ) -> None:
        """Create combined search result snippets from Oxylabs SERPResponse data search_information tag."""

        combined_search_result_tags = [
            ("organic", "Organic Results"),
            ("organic_videos", "Organic Videos"),
            ("paid", "Paid Results"),
            ("featured_snipped", "Feature Snipped"),
            ("top_stories", "Top Stories"),
            ("finance", "Finance"),
            ("sports_games", "Sports Games"),
            ("twitter", "Twitter"),
            ("discussions_and_forums", "Discussions and Forums"),
            ("images", "Images"),
            ("videos", "Videos"),
            ("video_box", "Video box"),
        ]
        self.process_tags(
            combined_search_result_snippets,
            combined_search_result_tags,
            results,
            "Combined Search Results",
        )

    def _create_product_information_snippets(
        self, results: dict, product_information_snippets: list
    ) -> None:
        """Create product information snippets from Oxylabs SERPResponse data search_information tag."""

        product_information_tags = [
            ("popular_products", "Popular Products"),
            ("pla", "Product Listing Ads (PLA)"),
        ]
        self.process_tags(
            product_information_snippets,
            product_information_tags,
            results,
            "Product Information",
        )

    def _create_local_information_snippets(
        self, results: dict, local_information_snippets: list
    ) -> None:
        """Create local group information snippets from Oxylabs SERPResponse data search_information tag."""

        local_information_tags = [
            ("top_sights", "Top Sights"),
            ("flights", "Flights"),
            ("hotels", "Hotels"),
            ("local_pack", "Local Pack"),
            ("local_service_ads", "Local Service Ads"),
            ("jobs", "Jobs"),
        ]
        self.process_tags(
            local_information_snippets,
            local_information_tags,
            results,
            "Local Information",
        )

    def _create_search_information_snippets(
        self, results: dict, search_information_snippets: list
    ) -> None:
        """Create search information snippets from Oxylabs SERPResponse data search_information tag."""

        search_information_tags = [
            ("search_information", "Search Information"),
            ("related_searches", "Related Searches"),
            ("related_searches_categorized", "Related Searches Categorized"),
            ("related_questions", "Related Questions"),
        ]
        self.process_tags(
            search_information_snippets,
            search_information_tags,
            results,
            "Search Information",
        )
