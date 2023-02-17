"""Loader that loads HN."""
from typing import List
from bs4 import BeautifulSoup as soup
from langchain.docstore.document import Document
from langchain.document_loaders.web_base import WebBaseLoader


class HNLoader(WebBaseLoader):
    """Loader that loads Hacker News data from either main page results or the comments page."""

    def __init__(self, web_path: str):
        self.web_path = web_path

    def load(self) -> List[Document]:
        """Get important HN webpage information.

        Components are:
            - title
            - content
            - source url,
            - time of post
            - author of the post
            - number of comments
            - rank of the post
        """
        soup = self.scrape()
        if ("item" in self.web_path):
            return self.load_comments(soup)
        else:
            return self.load_results(soup)

    def load_comments(self, soup: soup) -> List[Document]:
        """Load comments from a HN post."""
        comments = soup.select("tr[class='athing comtr']")
        title = soup.select_one("tr[id='pagespace']").get("title")
        documents = []
        for comment in comments:
            text = comment.text.strip()
            metadata = {"source": self.web_path, "title": title}
            documents.append(Document(page_content=text, metadata=metadata))
        return documents

    def load_results(self, soup: soup) -> List[Document]:
        """Load items from an HN page."""
        items = soup.select("tr[class='athing']")
        documents = []
        for lineItem in items:
            ranking = lineItem.select_one("span[class='rank']").text
            link = lineItem.find("span", {"class": "titleline"}).find("a").get("href")
            title = lineItem.find("span", {"class": "titleline"}).text.strip()
            metadata = {"source": self.web_path, "title": title,
                        "link": link, "ranking": ranking}
            documents.append(Document(page_content=title, link=link,
                             ranking=ranking, metadata=metadata))
        return documents
