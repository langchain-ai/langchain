"""Chain that calls Oxylabs API for Google Search."""

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from langchain_core.utils import get_from_dict_or_env
from pydantic import BaseModel, ConfigDict, Field, model_validator

RESULT_CATEGORIES = [
    "knowledge_graph",
    "combined_search_result",
    "product_information",
    "local_information",
    "search_information",
]

IMAGE_BINARY_CONTENT_ARRAY_ATTRIBUTE = "images"
BINARY_CONTENT_REPLACEMENT = "Redacted base64 image string..."


@dataclass
class ResponseElement:
    tag: str
    display_tag: str
    path_: str
    python_type: str
    parent: Optional["ResponseElement"]


def _get_default_excluded_result_attributes() -> List[str]:
    return ["pos_overall"]


def _get_default_image_content_attributes() -> List[str]:
    return ["image_data", "data"]


def _get_default_params() -> dict:
    """Provide default parameters for OxylabsSearchAPIWrapper.

    Returns:
        dict: Default parameters, including the following keys:
            - source (str): Source of the search engine, e.g., "google_search".
            - user_agent_type (str): User agent type, e.g.,
                "desktop". Can be set using values from `oxylabs.utils.types`.
            - render (str): Render type for the search results, e.g.,
                "html". Can be set using values from `oxylabs.utils.types`.
            - domain (str): Domain for the search engine, e.g., "com".
            - start_page (int): Starting page number for search results.
            - pages (int): Number of pages to retrieve.
            - limit (int): Maximum number of results to return.
            - parse (bool): Whether to enable result parsing into an object.
            - locale (str): Locale or location for the search.
            - geo_location (str): Geographic location for the search.
            - parsing_instructions (dict): Additional instructions for parsing
                the search results.
            - context (list): Search context information.
            - request_timeout (int): Timeout for the Oxylabs service, in seconds.
            - result_categories (list): Specifies the desired categories for the
                results from the available:
                    `knowledge_graph` which includes Oxylabs results categories:
                       `knowledge`, `recipes`, `carousel`, `apps`.
                    `combined_search_result` which includes Oxylabs results categories:
                       `organic`, `paid`, `organic_videos`, `featured_snippet`,
                       `top_stories`, `finance`, `sports_games`, `twitter`,
                       `discussions_and_forums`, `images`, `videos`, `video_box`.
                    `product_information` which includes Oxylabs results categories:
                       `popular_products`, `pla`.
                    `local_information` which includes Oxylabs results categories:
                       `top_sights`, `flights`, `hotels`, `local_pack`,
                       `local_service_ads`, `jobs`.
                    `search_information` which includes Oxylabs results categories:
                       `search_information`, `related_searches`,
                       `related_searches_categorized`, `related_questions`.
                The list preserves the order of the categories,
                allowing prioritized filtering. If left empty (default),
                results are returned from all available categories without filtering.
    """

    return {
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
        "result_categories": [],
    }


class OxylabsSearchAPIWrapper(BaseModel):
    """Wrapper class for OxylabsSearchAPI.

    Example:
        ```python
        from langchain_community.utilities import OxylabsSearchAPIWrapper

        oxylabs_api = OxylabsSearchAPIWrapper(
            "oxylabs_username": <OXYLABS_USERNAME>,
            "oxylabs_password": <OXYLABS_PASSWORD>,
            params={
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
                "result_categories": [],
            }
        )
        search_query = "Oxylabs"
        result = oxylabs_api.run(search_query)
        print(result)
        ```
    """

    include_binary_image_data: Optional[bool] = Field(default=False)
    parsing_recursion_depth: int = Field(default=5)

    search_engine: Any = None
    params: dict = Field(default_factory=_get_default_params)
    result_categories: Optional[list] = Field(default=[])

    excluded_result_attributes: List[str] = Field(
        default_factory=_get_default_excluded_result_attributes
    )
    image_binary_content_attributes: List[str] = Field(
        default_factory=_get_default_image_content_attributes
    )
    image_binary_content_array_attribute: str = Field(
        default=IMAGE_BINARY_CONTENT_ARRAY_ATTRIBUTE
    )
    binary_content_replacement: str = Field(default=BINARY_CONTENT_REPLACEMENT)

    oxylabs_username: Optional[str] = None
    oxylabs_password: Optional[str] = None
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """
        Validate that oxylabs username,
        password and python package exists in environment.
        """

        default_params = _get_default_params()
        default_params.update(**values.get("params", {}))
        current_params = default_params
        oxylabs_username = get_from_dict_or_env(
            values, "oxylabs_username", "OXYLABS_USERNAME"
        )
        oxylabs_password = get_from_dict_or_env(
            values, "oxylabs_password", "OXYLABS_PASSWORD"
        )
        if not (oxylabs_username and oxylabs_password):
            raise RuntimeError(
                "Please set up environment variables:"
                " `OXYLABS_USERNAME` and `OXYLABS_PASSWORD`"
                " or provide credentials in the parameters"
                " as `oxylabs_username` and `oxylabs_password`."
            )

        formed_values: Dict[str, Any] = dict()
        formed_values["oxylabs_username"] = oxylabs_username
        formed_values["oxylabs_password"] = oxylabs_password
        formed_values["params"] = dict()
        formed_values["params"] = dict(current_params)

        formed_values["include_binary_image_data"] = values.get(
            "include_binary_image_data", False
        )
        formed_values["parsing_recursion_depth"] = values.get(
            "parsing_recursion_depth", 5
        )

        if "result_categories" in formed_values["params"]:
            validated_categories = cls.validate_response_categories(
                formed_values["params"]["result_categories"]
            )
            if validated_categories:
                formed_values["result_categories"] = validated_categories

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
            source_ = formed_values["params"]["source"]
            source_provider_map = {
                "google": ["google_search"],
            }
            engine_ = ""
            for engine, sources_ in source_provider_map.items():
                if source_ in sources_:
                    engine_ = engine

                    break

            if engine_:
                oxylabs_realtime_client_by_provider = getattr(
                    oxylabs_realtime_client.serp, engine_
                )
                formed_values["search_engine"] = oxylabs_realtime_client_by_provider
            else:
                supported_sources = ", ".join(sum(source_provider_map.values(), []))
                raise NotImplementedError(
                    f"Source: `{source_}` is not supported."
                    f" Supported  sources: {supported_sources}"
                )

        except NotImplementedError as exc:
            raise NotImplementedError(f"{exc}")

        except Exception as exc:
            raise RuntimeError(f"Unknown Oxylabs Python SDK integration error: {exc}")

        return formed_values

    @staticmethod
    def validate_response_categories(result_categories: list) -> list:
        validated_categories = []
        for result_category in result_categories:
            if result_category in RESULT_CATEGORIES:
                validated_categories.append(result_category)

        return validated_categories

    async def arun(self, query: str, **kwargs: Any) -> str:
        """
        Run query through OxylabsSearchAPI and parse result async.
        """
        return self._process_response(await self.aresults(query, **kwargs), **kwargs)

    def run(self, query: str, **kwargs: Any) -> str:
        """
        Run query through OxylabsSearchAPI and parse result.
        """
        return self._process_response(self.results(query, **kwargs), **kwargs)

    def results(self, query: str, **kwargs: Any) -> List[Dict[str, Any]]:
        """
        Run query through Oxylabs Web Scrapper API and return SERPResponse object.
        """
        params_ = self.get_params(**kwargs)
        search_client = self.search_engine
        search_result = search_client.scrape_search(query, **params_)

        try:
            validated_responses = self._validate_response(search_result)

        except RuntimeError:
            return list()

        return validated_responses

    async def aresults(self, query: str, **kwargs: Any) -> List[Dict[str, Any]]:
        """
        Run query through Oxylabs Web Scrapper API and return SERPResponse object async.
        """
        params_ = self.get_params(**kwargs)

        search_client = self.search_engine
        search_result = await asyncio.to_thread(
            search_client.scrape_search,
            query,
            **params_,
        )

        try:
            validated_responses = self._validate_response(search_result)

        except RuntimeError:
            return list()

        return validated_responses

    def get_params(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Get default configuration parameters for OxylabsSearchAPI for scrape_search().
        """
        wrapper_params_ = ["result_categories"]

        _params = {
            f"{p_key}": self.params[p_key]
            for p_key in self.params
            if self.params[p_key] and p_key not in wrapper_params_
        }

        for key, value in kwargs.items():
            if key in _params:
                _params[key] = value

        return _params

    def _validate_response(self, response: Any) -> List[Dict[Any, Any]]:
        """
        Validate Oxylabs SERPResponse format and unpack data.
        """
        validated_results = list()
        try:
            result_pages = response.raw["results"]
            if not isinstance(result_pages, list) or not result_pages:
                raise ValueError("No results returned!")

            for result_page in result_pages:
                result_page = dict(result_page)
                content = result_page["content"]
                if not isinstance(content, dict):
                    raise ValueError(
                        "Result `content` format error,"
                        " try setting parameter `parse` to True"
                    )

                unpacked_results = content["results"]

                if not isinstance(unpacked_results, dict):
                    raise ValueError("Response format Error!")

                if unpacked_results:
                    validated_results.append(unpacked_results)

            return validated_results

        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Response Validation Error: {str(exc)}")

    def _process_response(self, res: Any, **kwargs: Any) -> str:
        """
        Process Oxylabs SERPResponse and serialize search results to string.
        """

        result_ = "No good search result found"

        result_category_processing_map = {
            "knowledge_graph": self._create_knowledge_graph_snippets,
            "combined_search_result": self._create_combined_search_result_snippets,
            "product_information": self._create_product_information_snippets,
            "local_information": self._create_local_information_snippets,
            "search_information": self._create_search_information_snippets,
        }

        snippets: List[str] = list()
        validated_categories = self.validate_response_categories(
            kwargs.get("result_categories", [])
        )
        result_categories_ = validated_categories or self.result_categories or []

        for nr_, validated_response in enumerate(res):
            if result_categories_:
                for result_category in result_categories_:
                    result_category_processing_map[result_category](
                        validated_response, snippets
                    )
            else:
                for result_category in result_category_processing_map:
                    result_category_processing_map[result_category](
                        validated_response, snippets
                    )

        # Combine all snippets
        if snippets:
            result_ = "\n\n".join(snippets)

        return result_

    def recursive_snippet_collector(
        self,
        target_structure: Any,
        max_depth: int,
        current_depth: int,
        parent_: ResponseElement,
    ) -> str:
        target_snippets: List[str] = list()

        padding_multiplier = current_depth + 1
        recursion_padding = "  " * padding_multiplier

        if current_depth >= max_depth:
            return "\n".join(target_snippets)

        if isinstance(target_structure, (str, float, int)):
            self.recursion_process_simple_types(
                parent_, recursion_padding, target_snippets, target_structure
            )

        elif isinstance(target_structure, dict):
            self.recursion_process_dict(
                current_depth,
                max_depth,
                parent_,
                recursion_padding,
                target_snippets,
                target_structure,
            )

        elif isinstance(target_structure, (list, tuple)):
            self.recursion_process_array(
                current_depth,
                max_depth,
                parent_,
                recursion_padding,
                target_snippets,
                target_structure,
            )

        return "\n".join(target_snippets)

    def recursion_process_array(
        self,
        current_depth: int,
        max_depth: int,
        parent_: ResponseElement,
        recursion_padding: str,
        target_snippets: list,
        target_structure: Any,
    ) -> None:
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

    def recursion_process_dict(
        self,
        current_depth: int,
        max_depth: int,
        parent_: ResponseElement,
        recursion_padding: str,
        target_snippets: list,
        target_structure: Any,
    ) -> None:
        if target_structure:
            target_snippets.append(
                f"{recursion_padding}{parent_.display_tag.upper()}: "
            )
            for key_, value_ in target_structure.items():
                if isinstance(value_, dict):
                    if value_:
                        target_snippets.append(f"{recursion_padding}{key_.upper()}: ")
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
                                        path_=f"{parent_.path_.upper()}"
                                        f"-{key_.upper()}-ITEM-{nr_ + 1}",
                                        tag=key_.upper(),
                                        display_tag=f"{key_.upper()}-ITEM-{nr_ + 1}",
                                        python_type=str(type(value_)),
                                        parent=parent_,
                                    ),
                                )
                            )

                elif isinstance(value_, (str, float, int)):
                    if value_:
                        if (
                            key_ in self.image_binary_content_attributes
                            and not self.include_binary_image_data
                        ):
                            value_ = self.binary_content_replacement

                        if key_ not in self.excluded_result_attributes:
                            target_snippets.append(
                                f"{recursion_padding}{key_.upper()}: {str(value_)}"
                            )

    def recursion_process_simple_types(
        self,
        parent_: ResponseElement,
        recursion_padding: str,
        target_snippets: list,
        target_structure: Any,
    ) -> None:
        if target_structure:
            if parent_.python_type == str(type(list())):
                if (
                    self.image_binary_content_array_attribute.upper()
                    in parent_.path_.split("-")[-3:]
                    or parent_.tag.lower() in self.image_binary_content_attributes
                ) and not self.include_binary_image_data:
                    target_structure = self.binary_content_replacement

                target_snippets.append(
                    f"{recursion_padding}{parent_.display_tag}: {str(target_structure)}"
                )

            elif parent_.python_type == str(type(dict())):
                if (
                    parent_.tag.lower() in self.image_binary_content_attributes
                    and not self.include_binary_image_data
                ):
                    target_structure = self.binary_content_replacement

                if parent_.tag.lower() not in self.excluded_result_attributes:
                    target_snippets.append(
                        f"{recursion_padding}{parent_.display_tag}:"
                        f" {str(target_structure)}"
                    )

    def process_tags(
        self, snippets_: list, tags_: list, results: dict, group_name: str = ""
    ) -> None:
        check_tags = [tag_[0] in results for tag_ in tags_]
        if any(check_tags):
            for tag in tags_:
                tag_content = results.get(tag[0], {}) or {}
                if tag_content:
                    collected_snippets = self.recursive_snippet_collector(
                        tag_content,
                        max_depth=self.parsing_recursion_depth,
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
        """
        Create knowledge graph snippets
        from Oxylabs SERPResponse data search_information tag.
        """

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
        """
        Create combined search result snippets
        from Oxylabs SERPResponse data search_information tag.
        """

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
        """
        Create product information snippets from
        Oxylabs SERPResponse data search_information tag.
        """

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
        """
        Create local group information snippets
        from Oxylabs SERPResponse data search_information tag.
        """

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
        """
        Create search information snippets
        from Oxylabs SERPResponse data search_information tag.
        """

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
