import asyncio
import logging
from base64 import b64decode
from typing import Any, Dict, Iterator, List, Literal, Optional

from langchain_core.documents import Document
from langchain_core.utils import get_from_env

from langchain_community.document_loaders.base import BaseLoader

logger = logging.getLogger(__name__)


class ZyteURLLoader(BaseLoader):
    """Load text from URLs using `Zyte api`.

    Args:
        urls: URLs to load. Each is loaded into its own document.
        api_key: Zyte API key.
        mode: Determines how the text is extracted for the page content.
            It can take one of the following values: 'html', 'html-text', 'article'
        continue_on_failure: If True, continue loading other URLs if one fails.
        **download_kwargs: Any additional download arguments to pass for download.
            See: https://docs.zyte.com/zyte-api/usage/reference.html

    Example:
        .. code-block:: python

            from langchain_community.document_loaders import ZyteURLLoader

            loader = ZyteURLLoader(
                urls=["<url-1>", "<url-2>"],
            )
            docs = loader.load()

    Zyte-API reference:
        https://www.zyte.com/zyte-api/

    """

    def __init__(
        self,
        urls: List[str],
        api_key: Optional[str],
        mode: Literal["article", "html", "html-text"] = "article",
        continue_on_failure: bool = False,
        download_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize with file path."""
        try:
            from zyte_api import AsyncZyteAPI, ZyteAPI
            from zyte_api.utils import USER_AGENT as PYTHON_ZYTE_API_USER_AGENT

        except ImportError:
            raise ImportError(
                "zyte-api package not found, please install it with "
                "`pip install zyte-api`"
            )
        if mode not in ("article", "html", "html-text"):
            raise ValueError(
                f"Unrecognized mode '{mode}'. Expected one of "
                f"'article', 'html', 'html-text'."
            )

        api_key = api_key or get_from_env("api_key", "ZYTE_API_KEY")
        user_agent = f"langchain-zyte-api/{PYTHON_ZYTE_API_USER_AGENT}"
        self.urls = urls
        self.mode = mode
        self.download_kwargs = download_kwargs or {}
        self.html_option = self._zyte_html_option()
        self.continue_on_failure = continue_on_failure
        self.client = ZyteAPI(api_key=api_key, user_agent=user_agent)
        self.client_async = AsyncZyteAPI(api_key=api_key, user_agent=user_agent)

    def _zyte_html_option(self) -> str:
        if "browserHtml" in self.download_kwargs:
            return "browserHtml"
        return "httpResponseBody"

    def load(self) -> List[Document]:
        iter = self.lazy_load()
        return list(iter)

    def _get_article(self, page: Dict) -> str:
        content = page["article"]["headline"] + "\n\n" + page["article"]["articleBody"]
        return content

    def _zyte_request_params(self, url: str) -> dict:
        request_params: Dict[str, Any] = {"url": url}
        if self.mode == "article":
            request_params.update({"article": True})

        if self.mode in ("html", "html-text"):
            request_params.update({self.html_option: True})

        if self.download_kwargs:
            request_params.update(self.download_kwargs)
        return request_params

    def lazy_load(self) -> Iterator[Document]:
        queries = [self._zyte_request_params(url) for url in self.urls]
        with self.client.session() as session:
            for i, response in enumerate(session.iter(queries)):
                if not isinstance(response, dict):
                    url = queries[i]["url"]
                    if self.continue_on_failure:
                        logger.warning(
                            f"Error {response} while fetching url {url}, "
                            f"skipping because continue_on_failure is True"
                        )
                        continue
                    else:
                        logger.exception(
                            f"Error fetching {url} and aborting, use "
                            f"continue_on_failure=True to continue loading "
                            f"urls after encountering an error."
                        )
                        raise response

                content = self._get_content(response)
                yield Document(page_content=content, metadata={"url": response["url"]})

    async def fetch_items(self) -> List:
        results = []
        queries = [self._zyte_request_params(url) for url in self.urls]
        async with self.client_async.session() as session:
            for i, future in enumerate(session.iter(queries)):
                try:
                    result = await future
                    results.append(result)
                except Exception as e:
                    url = queries[i]["url"]
                    if self.continue_on_failure:
                        logger.warning(
                            f"Error {e} while fetching url {url}, "
                            f"skipping because continue_on_failure is True"
                        )
                        continue
                    else:
                        logger.exception(
                            f"Error fetching {url} and aborting, use "
                            f"continue_on_failure=True to continue loading "
                            f"urls after encountering an error."
                        )
                        raise e
        return results

    def _get_content(self, response: Dict) -> str:
        if self.mode == "html-text":
            try:
                from html2text import html2text

            except ImportError:
                raise ImportError(
                    "html2text package not found, please install it with "
                    "`pip install html2text`"
                )
        if self.mode in ("html", "html-text"):
            content = response[self.html_option]

            if self.html_option == "httpResponseBody":
                content = b64decode(content).decode()

            if self.mode == "html-text":
                content = html2text(content)
        elif self.mode == "article":
            content = self._get_article(response)
        return content

    def aload(self) -> List[Document]:
        docs = []
        responses = asyncio.run(self.fetch_items())
        for response in responses:
            content = self._get_content(response)
            doc = Document(page_content=content, metadata={"url": response["url"]})
            docs.append(doc)
        return docs
