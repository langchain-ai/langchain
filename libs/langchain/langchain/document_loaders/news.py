"""Loader that uses unstructured to load HTML files."""
import logging
from typing import Any, List

from newspaper import Article

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

logger = logging.getLogger(__name__)


class NewsURLLoader(BaseLoader):
    """Loader that uses newspaper to load news articles from URLs.

    Examples
    --------
    from langchain.document_loaders import NewsURLLoader

    loader = NewsURLLoader(
        urls=["<url-1>", "<url-2>"],
    )
    docs = loader.load()

    References
    ----------
    https://newspaper.readthedocs.io/en/latest/
    """

    def __init__(
        self,
        urls: List[str],
        text_mode=True,
        nlp=False,
        continue_on_failure: bool = True,
        show_progress_bar: bool = False,
        **newspaper_kwargs: Any,
    ):
        """Initialize with file path."""
        try:
            import newspaper  # noqa:F401

            self.__version = newspaper.__version__
        except ImportError:
            raise ImportError(
                "newspaper package not found, please install it with "
                "`pip install newspaper3k`"
            )

        self.urls = urls
        self.text_mode = text_mode
        self.nlp = nlp
        self.continue_on_failure = continue_on_failure
        self.newspaper_kwargs = newspaper_kwargs
        self.show_progress_bar = show_progress_bar

    def load(self) -> List[Document]:

        docs: List[Document] = list()
        if self.show_progress_bar:
            try:
                from tqdm import tqdm
            except ImportError as e:
                raise ImportError(
                    "Package tqdm must be installed if show_progress_bar=True. "
                    "Please install with 'pip install tqdm' or set "
                    "show_progress_bar=False."
                ) from e

            urls = tqdm(self.urls)
        else:
            urls = self.urls

        for url in urls:
            try:
                article = Article(url, **self.newspaper_kwargs)
                article.download()
                article.parse()

                if self.nlp:
                    article.nlp()

            except Exception as e:
                if self.continue_on_failure:
                    logger.error(f"Error fetching or processing {url}, exception: {e}")
                    continue
                else:
                    raise e

            metadata = {
                "title": article.title if hasattr(article, "title") else "",
                "link": article.url
                if hasattr(article, "url")
                else article.canonical_link
                if hasattr(article, "canonical_link")
                else "",
                "authors": article.authors if hasattr(article, "authors") else "",
                "language": article.meta_lang if hasattr(article, "meta_lang") else "",
                "description": article.meta_description
                if hasattr(article, "meta_description")
                else "",
                "publish_date": article.publish_date
                if hasattr(article, "publish_date")
                else "",
            }

            if self.text_mode:
                content = article.text
            else:
                content = article.html

            if self.nlp:
                metadata["keywords"] = (
                    article.keywords if hasattr(article, "keywords") else []
                )
                metadata["summary"] = (
                    article.summary if hasattr(article, "summary") else ""
                )

            docs.append(Document(page_content=content, metadata=metadata))

        return docs
