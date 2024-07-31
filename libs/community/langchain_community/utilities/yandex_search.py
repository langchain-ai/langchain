from __future__ import annotations

import re
from typing import Any, Dict, List, Literal

from langchain_core.pydantic_v1 import BaseModel, Extra, SecretStr, root_validator
from langchain_core.utils import get_from_dict_or_env


class YandexSearchAPIWrapper(BaseModel):
    """Wrapper for Yandex Search API client."""

    folder_id: SecretStr
    api_key: SecretStr
    base_url: str = "https://yandex.ru"
    search_region_id: int = 225
    notification_language: str = "ru"
    sorting_rule: Literal["relevance", "document_update_time"] = "relevance"
    sorting_time_order: Literal["descending", "ascending"] = "descending"
    filtering_type: Literal["none", "moderate", "strict"] = "moderate"
    passages_count: int = 5
    grouping_method: Literal["flat", "deep"] = "deep"
    groups_on_page: int = 100
    docs_in_group: int = 1
    page: int = 0
    answer_fields: List[str] = ["url", "content"]
    passages_delimiter: str = "\n"

    @property
    def base_params(self) -> Dict:
        sorting_rule_map = {
            "relevance": "rlv",
            "document_update_time": "tm",
        }
        sorting_type = f"{sorting_rule_map[self.sorting_rule]}"
        if self.sorting_rule == "document_update_time":
            sorting_type += f".order={self.sorting_time_order}"

        if self.grouping_method == "deep":
            grouping_attr_param = "attr=d"
        else:
            grouping_attr_param = ""
        groups_on_page_param = f"groups-on-page={self.groups_on_page}"
        docs_in_group_param = f"docs-in-group={self.docs_in_group}"
        result_grouping_parameters = f"{groups_on_page_param}.{docs_in_group_param}"
        if grouping_attr_param:
            result_grouping_parameters = (
                f"{grouping_attr_param}.{result_grouping_parameters}"
            )

        params = {
            "folderid": self.folder_id.get_secret_value(),
            "apikey": self.api_key.get_secret_value(),
            "lr": self.search_region_id,
            "l10n": self.notification_language,
            "sortby": sorting_type,
            "filter": self.filtering_type,
            "maxpassages": self.passages_count,
            "groupby": result_grouping_parameters,
            "page": self.page,
        }

        return params

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and folder id exist in environment."""
        api_key = get_from_dict_or_env(values, "api_key", "YANDEX_API_KEY")
        values["api_key"] = api_key

        folder_id = get_from_dict_or_env(values, "folder_id", "YANDEX_FOLDER_ID")
        values["folder_id"] = folder_id
        return values

    def _parse_results(self, results: str) -> List[Dict[str, Any]]:
        try:
            from parsel import Selector
        except ImportError as e:
            raise ImportError(
                "Could not import parsel python package. "
                "Please install it with `pip install parsel`."
            ) from e

        selector = Selector(text=results)
        tag_pattern = re.compile(r"</?.*?>")
        docs = []

        error = selector.xpath("//error/text()").get()

        if error:
            raise RuntimeError(error)

        for doc in selector.xpath("//doc"):
            doc_id = doc.xpath("./@id").get()

            if not doc_id:
                continue

            title = doc.xpath("./title").get(default="")
            if title:
                title = tag_pattern.sub("", title)

            headline = doc.xpath("./headline").get(default="")
            if headline:
                headline = tag_pattern.sub("", headline)

            passages = doc.xpath("./passages//passage").getall()
            if passages:
                passages = [tag_pattern.sub("", passage) for passage in passages]

            modified_at = doc.xpath("./modtime/text()").get()

            url = doc.xpath("./url/text()").get()
            saved_copy_url = doc.xpath("./saved-copy-url/text()").get()

            if passages:
                text = self.passages_delimiter.join(passages)
                text_type = "passages"
            else:
                text = headline
                text_type = "headline"

            docs.append(
                {
                    "modified_at": modified_at,
                    "title": title,
                    "headline": headline,
                    "passages": passages,
                    "url": url,
                    "saved_copy_url": saved_copy_url,
                    "content": text,
                    "content_type": text_type,
                }
            )

        answer_docs = []

        for document in docs:
            answer_docs.append(
                {
                    field: document[field]
                    for field in self.answer_fields
                    if document.get(field)
                }
            )

        return answer_docs

    def _search(
        self,
        query: str,
    ) -> str:
        params = {
            "query": query,
            **self.base_params,
        }

        try:
            import httpx
        except ImportError as e:
            raise ImportError(
                "Could not import httpx python package. "
                "Please install it with `pip install httpx`."
            ) from e

        with httpx.Client(base_url=self.base_url) as client:
            response = client.get("/search/xml", params=params)

        response.raise_for_status()

        return response.text

    async def _asearch(
        self,
        query: str,
    ) -> str:
        params = {
            "query": query,
            **self.base_params,
        }

        try:
            import httpx
        except ImportError as e:
            raise ImportError(
                "Could not import httpx python package. "
                "Please install it with `pip install httpx`."
            ) from e

        async with httpx.AsyncClient(base_url=self.base_url) as client:
            response = await client.get("/search/xml", params=params)

        response.raise_for_status()

        return response.text

    def raw_results(
        self,
        query: str,
    ) -> str:
        results = self._search(query)
        return results

    def results(
        self,
        query: str,
    ) -> List[Dict]:
        results = self.raw_results(query)
        return self._parse_results(results)

    async def raw_results_async(
        self,
        query: str,
    ) -> str:
        results = await self._asearch(query)
        return results

    async def results_async(
        self,
        query: str,
    ) -> List[Dict]:
        results = await self.raw_results_async(query)
        return self._parse_results(results)
