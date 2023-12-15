import logging
from typing import Any, Iterator, List, Optional, Sequence

from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders.news import NewsURLLoader

logger = logging.getLogger(__name__)


class RSSFeedLoader(BaseLoader):
    """Load news articles from `RSS` feeds using `Unstructured`.

    Args:
        urls: URLs for RSS feeds to load. Each articles in the feed is loaded into its own document.
        opml: OPML file to load feed urls from. Only one of urls or opml should be provided.  The value
        can be a URL string, or OPML markup contents as byte or string.
        continue_on_failure: If True, continue loading documents even if
            loading fails for a particular URL.
        show_progress_bar: If True, use tqdm to show a loading progress bar. Requires
            tqdm to be installed, ``pip install tqdm``.
        **newsloader_kwargs: Any additional named arguments to pass to
            NewsURLLoader.

    Example:
        .. code-block:: python

            from langchain_community.document_loaders import RSSFeedLoader

            loader = RSSFeedLoader(
                urls=["<url-1>", "<url-2>"],
            )
            docs = loader.load()

    The loader uses feedparser to parse RSS feeds.  The feedparser library is not installed by default so you should
    install it if using this loader:
    https://pythonhosted.org/feedparser/

    If you use OPML, you should also install listparser:
    https://pythonhosted.org/listparser/

    Finally, newspaper is used to process each article:
    https://newspaper.readthedocs.io/en/latest/
    """  # noqa: E501

    def __init__(
        self,
        urls: Optional[Sequence[str]] = None,
        opml: Optional[str] = None,
        continue_on_failure: bool = True,
        show_progress_bar: bool = False,
        **newsloader_kwargs: Any,
    ) -> None:
        """Initialize with urls or OPML."""
        if (urls is None) == (
            opml is None
        ):  # This is True if both are None or neither is None
            raise ValueError(
                "Provide either the urls or the opml argument, but not both."
            )
        self.urls = urls
        self.opml = opml
        self.continue_on_failure = continue_on_failure
        self.show_progress_bar = show_progress_bar
        self.newsloader_kwargs = newsloader_kwargs

    def load(self) -> List[Document]:
        iter = self.lazy_load()
        if self.show_progress_bar:
            try:
                from tqdm import tqdm
            except ImportError as e:
                raise ImportError(
                    "Package tqdm must be installed if show_progress_bar=True. "
                    "Please install with 'pip install tqdm' or set "
                    "show_progress_bar=False."
                ) from e
            iter = tqdm(iter)
        return list(iter)

    @property
    def _get_urls(self) -> Sequence[str]:
        if self.urls:
            return self.urls
        try:
            import listparser
        except ImportError as e:
            raise ImportError(
                "Package listparser must be installed if the opml arg is used. "
                "Please install with 'pip install listparser' or use the "
                "urls arg instead."
            ) from e
        rss = listparser.parse(self.opml)
        return [feed.url for feed in rss.feeds]

    def lazy_load(self) -> Iterator[Document]:
        try:
            import feedparser  # noqa:F401
        except ImportError:
            raise ImportError(
                "feedparser package not found, please install it with "
                "`pip install feedparser`"
            )

        for url in self._get_urls:
            try:
                feed = feedparser.parse(url)
                if getattr(feed, "bozo", False):
                    raise ValueError(
                        f"Error fetching {url}, exception: {feed.bozo_exception}"
                    )
            except Exception as e:
                if self.continue_on_failure:
                    logger.error(f"Error fetching {url}, exception: {e}")
                    continue
                else:
                    raise e
            try:
                for entry in feed.entries:
                    loader = NewsURLLoader(
                        urls=[entry.link],
                        **self.newsloader_kwargs,
                    )
                    article = loader.load()[0]
                    article.metadata["feed"] = url
                    yield article
            except Exception as e:
                if self.continue_on_failure:
                    logger.error(f"Error processing entry {entry.link}, exception: {e}")
                    continue
                else:
                    raise e
