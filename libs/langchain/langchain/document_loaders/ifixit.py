"""Loads iFixit data."""
from typing import List, Optional

import requests

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.document_loaders.web_base import WebBaseLoader

IFIXIT_BASE_URL = "https://www.ifixit.com/api/2.0"


class IFixitLoader(BaseLoader):
    """Load iFixit repair guides, device wikis and answers.

    iFixit is the largest, open repair community on the web. The site contains nearly
    100k repair manuals, 200k Questions & Answers on 42k devices, and all the data is
    licensed under CC-BY.

    This loader will allow you to download the text of a repair guide, text of Q&A's
    and wikis from devices on iFixit using their open APIs and web scraping.
    """

    def __init__(self, web_path: str):
        """Initialize with a web path."""
        if not web_path.startswith("https://www.ifixit.com"):
            raise ValueError("web path must start with 'https://www.ifixit.com'")

        path = web_path.replace("https://www.ifixit.com", "")

        allowed_paths = ["/Device", "/Guide", "/Answers", "/Teardown"]

        """ TODO: Add /Wiki """
        if not any(path.startswith(allowed_path) for allowed_path in allowed_paths):
            raise ValueError(
                "web path must start with /Device, /Guide, /Teardown or /Answers"
            )

        pieces = [x for x in path.split("/") if x]

        """Teardowns are just guides by a different name"""
        self.page_type = pieces[0] if pieces[0] != "Teardown" else "Guide"

        if self.page_type == "Guide" or self.page_type == "Answers":
            self.id = pieces[2]
        else:
            self.id = pieces[1]

        self.web_path = web_path

    def load(self) -> List[Document]:
        if self.page_type == "Device":
            return self.load_device()
        elif self.page_type == "Guide" or self.page_type == "Teardown":
            return self.load_guide()
        elif self.page_type == "Answers":
            return self.load_questions_and_answers()
        else:
            raise ValueError("Unknown page type: " + self.page_type)

    @staticmethod
    def load_suggestions(query: str = "", doc_type: str = "all") -> List[Document]:
        """Load suggestions.

        Args:
            query: A query string
            doc_type: The type of document to search for. Can be one of "all",
              "device", "guide", "teardown", "answer", "wiki".

        Returns:

        """
        res = requests.get(
            IFIXIT_BASE_URL + "/suggest/" + query + "?doctypes=" + doc_type
        )

        if res.status_code != 200:
            raise ValueError(
                'Could not load suggestions for "' + query + '"\n' + res.json()
            )

        data = res.json()

        results = data["results"]
        output = []

        for result in results:
            try:
                loader = IFixitLoader(result["url"])
                if loader.page_type == "Device":
                    output += loader.load_device(include_guides=False)
                else:
                    output += loader.load()
            except ValueError:
                continue

        return output

    def load_questions_and_answers(
        self, url_override: Optional[str] = None
    ) -> List[Document]:
        """Load a list of questions and answers.

        Args:
            url_override: A URL to override the default URL.

        Returns: List[Document]

        """
        loader = WebBaseLoader(self.web_path if url_override is None else url_override)
        soup = loader.scrape()

        output = []

        title = soup.find("h1", "post-title").text

        output.append("# " + title)
        output.append(soup.select_one(".post-content .post-text").text.strip())

        answersHeader = soup.find("div", "post-answers-header")
        if answersHeader:
            output.append("\n## " + answersHeader.text.strip())

        for answer in soup.select(".js-answers-list .post.post-answer"):
            if answer.has_attr("itemprop") and "acceptedAnswer" in answer["itemprop"]:
                output.append("\n### Accepted Answer")
            elif "post-helpful" in answer["class"]:
                output.append("\n### Most Helpful Answer")
            else:
                output.append("\n### Other Answer")

            output += [
                a.text.strip() for a in answer.select(".post-content .post-text")
            ]
            output.append("\n")

        text = "\n".join(output).strip()

        metadata = {"source": self.web_path, "title": title}

        return [Document(page_content=text, metadata=metadata)]

    def load_device(
        self, url_override: Optional[str] = None, include_guides: bool = True
    ) -> List[Document]:
        """Loads a device

        Args:
            url_override: A URL to override the default URL.
            include_guides: Whether to include guides linked to from the device.
              Defaults to True.

        Returns:

        """
        documents = []
        if url_override is None:
            url = IFIXIT_BASE_URL + "/wikis/CATEGORY/" + self.id
        else:
            url = url_override

        res = requests.get(url)
        data = res.json()
        text = "\n".join(
            [
                data[key]
                for key in ["title", "description", "contents_raw"]
                if key in data
            ]
        ).strip()

        metadata = {"source": self.web_path, "title": data["title"]}
        documents.append(Document(page_content=text, metadata=metadata))

        if include_guides:
            """Load and return documents for each guide linked to from the device"""
            guide_urls = [guide["url"] for guide in data["guides"]]
            for guide_url in guide_urls:
                documents.append(IFixitLoader(guide_url).load()[0])

        return documents

    def load_guide(self, url_override: Optional[str] = None) -> List[Document]:
        """Load a guide

        Args:
            url_override: A URL to override the default URL.

        Returns: List[Document]

        """
        if url_override is None:
            url = IFIXIT_BASE_URL + "/guides/" + self.id
        else:
            url = url_override

        res = requests.get(url)

        if res.status_code != 200:
            raise ValueError(
                "Could not load guide: " + self.web_path + "\n" + res.json()
            )

        data = res.json()

        doc_parts = ["# " + data["title"], data["introduction_raw"]]

        doc_parts.append("\n\n###Tools Required:")
        if len(data["tools"]) == 0:
            doc_parts.append("\n - None")
        else:
            for tool in data["tools"]:
                doc_parts.append("\n - " + tool["text"])

        doc_parts.append("\n\n###Parts Required:")
        if len(data["parts"]) == 0:
            doc_parts.append("\n - None")
        else:
            for part in data["parts"]:
                doc_parts.append("\n - " + part["text"])

        for row in data["steps"]:
            doc_parts.append(
                "\n\n## "
                + (
                    row["title"]
                    if row["title"] != ""
                    else "Step {}".format(row["orderby"])
                )
            )

            for line in row["lines"]:
                doc_parts.append(line["text_raw"])

        doc_parts.append(data["conclusion_raw"])

        text = "\n".join(doc_parts)

        metadata = {"source": self.web_path, "title": data["title"]}

        return [Document(page_content=text, metadata=metadata)]
