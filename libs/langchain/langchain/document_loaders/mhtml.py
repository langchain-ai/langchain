"""Load MHTML files, enriching metadata with page title."""

import email
import logging
from typing import Dict, List, Union

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

logger = logging.getLogger(__name__)


class MHTMLLoader(BaseLoader):
    """Loader that uses beautiful soup to parse HTML files."""

    def __init__(
        self,
        file_path: str,
        open_encoding: Union[str, None] = None,
        bs_kwargs: Union[dict, None] = None,
        get_text_separator: str = "",
    ) -> None:
        """Initialise with path, and optionally, file encoding to use, and any kwargs
        to pass to the BeautifulSoup object.

        Args:
            file_path: Path to file to load.
            open_encoding: The encoding to use when opening the file.
            bs_kwargs: Any kwargs to pass to the BeautifulSoup object.
            get_text_separator: The separator to use when getting the text
                from the soup.
        """
        try:
            import bs4  # noqa:F401
        except ImportError:
            raise ImportError(
                "beautifulsoup4 package not found, please install it with "
                "`pip install beautifulsoup4`"
            )

        self.file_path = file_path
        self.open_encoding = open_encoding
        if bs_kwargs is None:
            bs_kwargs = {"features": "lxml"}
        self.bs_kwargs = bs_kwargs
        self.get_text_separator = get_text_separator

    def load(self) -> List[Document]:
        from bs4 import BeautifulSoup

        """Load MHTML document into document objects."""

        with open(self.file_path, "r", encoding=self.open_encoding) as f:
            message = email.message_from_string(f.read())
            parts = message.get_payload()

            if type(parts) is not list:
                parts = [message]

            for part in parts:
                if part.get_content_type() == "text/html":
                    html = part.get_payload(decode=True).decode()

                    soup = BeautifulSoup(html, **self.bs_kwargs)
                    text = soup.get_text(self.get_text_separator)

                    if soup.title:
                        title = str(soup.title.string)
                    else:
                        title = ""

                    metadata: Dict[str, Union[str, None]] = {
                        "source": self.file_path,
                        "title": title,
                    }
                    return [Document(page_content=text, metadata=metadata)]
        return []
