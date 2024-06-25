from __future__ import annotations

import re
import typing
from typing import Any, Dict, List

if typing.TYPE_CHECKING:
    import httpx


class YandexSearchAPIClient:
    def __init__(
        self,
        folderid: str,
        apikey: str,
        base_url: str = "https://yandex.ru",
        lr: int = 225,
        l10n: str = "ru",
        sortby: str = "rlv.order=descending",
        filter: str = "moderate",
        maxpassages: int = 5,
        groupby: str = "attr=d.mode=deep.groups-on-page=100.docs-in-group=3",
        page: int = 0,
    ) -> None:
        self.base_url = base_url
        self.folderid = folderid
        self.apikey = apikey
        self.lr = lr
        self.l10n = l10n
        self.sortby = sortby
        self.filter = filter
        self.maxpassages = maxpassages
        self.groupby = groupby
        self.page = page

    def search(self, query: str) -> List[Dict[str, Any]]:
        params: Dict[str, str | int] = {
            "folderid": self.folderid,
            "apikey": self.apikey,
            "query": query,
            "lr": self.lr,
            "l10n": self.l10n,
            "sortby": self.sortby,
            "filter": self.filter,
            "maxpassages": self.maxpassages,
            "groupby": self.groupby,
            "page": self.page,
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

        processed_response = self._process_response(response)
        return processed_response

    async def asearch(self, query: str) -> List[Dict[str, Any]]:
        params: Dict[str, str | int] = {
            "folderid": self.folderid,
            "apikey": self.apikey,
            "query": query,
            "lr": self.lr,
            "l10n": self.l10n,
            "sortby": self.sortby,
            "filter": self.filter,
            "maxpassages": self.maxpassages,
            "groupby": self.groupby,
            "page": self.page,
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

        processed_response = self._process_response(response)
        return processed_response

    @staticmethod
    def _process_response(response: httpx.Response) -> list[dict[str, Any]]:
        try:
            from parsel import Selector
        except ImportError as e:
            raise ImportError(
                "Could not import httpx python package. "
                "Please install it with `pip install parsel`."
            ) from e

        selector = Selector(text=response.text)
        tag_pattern = re.compile(r"</?[a-z]+>")
        docs = []

        error = selector.xpath("//error/text()").get()

        if error:
            raise RuntimeError(error)

        for doc in selector.xpath("//doc"):
            doc_id = doc.xpath("./@id").get(default="")

            # Answer for image search
            if not doc_id:
                continue

            title = doc.xpath("./title").get(default="")
            title = re.sub(tag_pattern, "", title)

            headline = doc.xpath("./headline").get(default="")
            headline = re.sub(tag_pattern, "", headline)

            passages = doc.xpath("./passages//passage").getall()
            passages = [re.sub(tag_pattern, "", passage) for passage in passages]

            modified_at = doc.xpath("./modtime/text()").get(default="")

            url = doc.xpath("./url/text()").get(default="")
            saved_copy_url = doc.xpath("./saved-copy-url/text()").get(default="")

            docs.append(
                {
                    "modified_at": modified_at,
                    "title": title,
                    "headline": headline,
                    "passages": passages,
                    "url": url,
                    "saved_copy_url": saved_copy_url,
                }
            )

        return docs
