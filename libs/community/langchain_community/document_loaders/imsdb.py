from typing import List

from langchain_core.documents import Document

from langchain_community.document_loaders.web_base import WebBaseLoader


class IMSDbLoader(WebBaseLoader):
    """Load `IMSDb` webpages."""

    def load(self) -> List[Document]:
        """Load webpage."""
        soup = self.scrape()
        text = soup.select_one("td[class='scrtext']").text
        metadata = {"source": self.web_path}
        return [Document(page_content=text, metadata=metadata)]
