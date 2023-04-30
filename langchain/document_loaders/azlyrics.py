"""Loader that loads AZLyrics."""
from typing import List

from langchain.docstore.document import Document
from langchain.document_loaders.web_base import WebBaseLoader


class AZLyricsLoader(WebBaseLoader):
    """Loader that loads AZLyrics webpages."""

    def load(self) -> List[Document]:
        """Load webpage."""
        soup = self.scrape()
        title = soup.title.text
        lyrics = soup.find_all("div", {"class": ""})[2].text
        text = title + lyrics
        metadata = {"source": self.web_path}
        return [Document(page_content=text, metadata=metadata)]
