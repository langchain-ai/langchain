from typing import List

from langchain_core.documents import Document

from langchain.document_loaders.web_base import WebBaseLoader


class AZLyricsLoader(WebBaseLoader):
    """Load `AZLyrics` webpages."""

    def load(self) -> List[Document]:
        """Load webpages into Documents."""
        soup = self.scrape()
        title = soup.title.text
        lyrics = soup.find_all("div", {"class": ""})[2].text
        text = title + lyrics
        metadata = {"source": self.web_path}
        return [Document(page_content=text, metadata=metadata)]
