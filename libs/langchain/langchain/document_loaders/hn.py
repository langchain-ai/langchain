from typing import Any, List

from langchain_core.documents import Document

from langchain.document_loaders.web_base import WebBaseLoader


class HNLoader(WebBaseLoader):
    """Load `Hacker News` data.

    It loads data from either main page results or the comments page."""

    def load(self) -> List[Document]:
        """Get important HN webpage information.

        HN webpage components are:
            - title
            - content
            - source url,
            - time of post
            - author of the post
            - number of comments
            - rank of the post
        """
        soup_info = self.scrape()
        if "item" in self.web_path:
            return self.load_comments(soup_info)
        else:
            return self.load_results(soup_info)

    def load_comments(self, soup_info: Any) -> List[Document]:
        """Load comments from a HN post."""
        comments = soup_info.select("tr[class='athing comtr']")
        title = soup_info.select_one("tr[id='pagespace']").get("title")
        return [
            Document(
                page_content=comment.text.strip(),
                metadata={"source": self.web_path, "title": title},
            )
            for comment in comments
        ]

    def load_results(self, soup: Any) -> List[Document]:
        """Load items from an HN page."""
        items = soup.select("tr[class='athing']")
        documents = []
        for lineItem in items:
            ranking = lineItem.select_one("span[class='rank']").text
            link = lineItem.find("span", {"class": "titleline"}).find("a").get("href")
            title = lineItem.find("span", {"class": "titleline"}).text.strip()
            metadata = {
                "source": self.web_path,
                "title": title,
                "link": link,
                "ranking": ranking,
            }
            documents.append(
                Document(
                    page_content=title, link=link, ranking=ranking, metadata=metadata
                )
            )
        return documents
