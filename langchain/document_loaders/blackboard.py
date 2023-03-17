"""Loader that loads all documents from a blackboard course."""
import contextlib
import re
from pathlib import Path
from typing import Any, List, Optional, Tuple
from urllib.parse import unquote, urlparse

from langchain.docstore.document import Document
from langchain.document_loaders.web_base import WebBaseLoader
from langchain.document_loaders import PyPDFLoader, DirectoryLoader


class BlackboardLoader(WebBaseLoader):
    """Loader that loads all documents from a blackboard course."""

    base_url: str
    folder_path: str
    load_all_recursively: bool

    def __init__(
        self,
        blackboard_course_url: str,
        basic_auth: Optional[Tuple[str, str]] = None,
        bbrouter: Optional[str] = None,
        cookies: Optional[dict] = None,
        load_all_recursively: bool = True,
    ):
        """Initialize with blackboard course url.

        The BbRouter cookie is required for most blackboard courses.
        """
        super().__init__(blackboard_course_url)
        # Get base url
        try:
            self.base_url = blackboard_course_url.split("/webapps/blackboard")[0]
        except IndexError:
            raise ValueError(
                "Invalid blackboard course url. "
                "Please provide a url that starts with "
                "https://<blackboard_url>/webapps/blackboard"
            )
        if basic_auth is not None:
            self.session.auth = basic_auth
        if cookies is not None:
            self.session.cookies.update(cookies)
        if bbrouter is not None:
            self.session.cookies.update({"BbRouter": bbrouter})
        self.load_all_recursively = load_all_recursively
        self.check_bs4()

    def check_bs4(self):
        try:
            import bs4  # noqa: F401
        except ImportError:
            raise ImportError(
                "BeautifulSoup4 is required for BlackboardLoader. "
                "Please install it with `pip install beautifulsoup4`."
            )

    def _scrape(self, url: str) -> Any:
        from bs4 import BeautifulSoup

        html_doc = self.session.get(url)
        soup = BeautifulSoup(html_doc.text, "html.parser")
        return soup

    def scrape(self) -> Any:
        """Scrape data from webpage and return it in BeautifulSoup format."""
        return self._scrape(self.web_path)

    def load(self) -> List[Document]:
        """Load documents."""
        if self.load_all_recursively:
            soup_info = self.scrape()
            self.folder_path = self._get_folder_path(soup_info)
            relative_paths = self._get_paths(soup_info)
            documents = []
            for path in relative_paths:
                url = self.base_url + path
                print(f"Fetching documents from {url}")
                soup_info = self._scrape(url)
                with contextlib.suppress(ValueError):
                    documents.extend(self._get_documents(soup_info))
            return documents
        else:
            print(f"Fetching documents from {self.web_path}")
            soup_info = self.scrape()
            self.folder_path = self._get_folder_path(soup_info)
            return self._get_documents(soup_info)

    def _get_documents(self, soup: Any) -> List[Document]:
        """Fetch content from page and return Documents."""
        from bs4 import Tag

        # Get content list
        content_list = soup.find("ul", {"class": "contentList"})
        if content_list is None:
            raise ValueError("No content list found.")
        # Get all attachments
        attachments = []
        for attachment in content_list.find_all("ul", {"class": "attachments"}):
            for link in attachment.find_all("a"):
                link: Tag
                href = link.get("href")
                # Only add if href is not None and the text contains .pdf
                if href is not None:
                    attachments.append(href)
        # Make sure the folder exists
        Path(self.folder_path).mkdir(parents=True, exist_ok=True)
        # Download all attachments
        for attachment in attachments:
            self.download(attachment)
        # Create the document loader
        loader = DirectoryLoader(
            path=Path(self.folder_path), glob="*.pdf", loader_cls=PyPDFLoader
        )
        # Load the documents
        documents = loader.load()
        # Return all documents
        return documents

    def _get_folder_path(self, soup: Any) -> str:
        """Get the folder path to save the documents in."""
        # Get the course name
        course_name = soup.find("span", {"id": "crumb_1"})
        if course_name is None:
            raise ValueError("No course name found.")
        course_name = course_name.text.strip()
        # Prepare the folder path
        course_name_clean = (
            unquote(course_name).replace(" ", "_").replace("/", "_").replace(":", "_")
        )
        # Get the folder path
        folder_path = Path(".") / course_name_clean
        return str(folder_path)

    def _get_paths(self, soup: Any) -> List[str]:
        """Get all relative paths in the navbar."""
        relative_paths = []
        course_menu = soup.find("ul", {"class": "courseMenu"})
        if course_menu is None:
            raise ValueError("No course menu found.")
        for link in course_menu.find_all("a"):
            href = link.get("href")
            if href is not None and href.startswith("/"):
                relative_paths.append(href)
        return relative_paths

    def download(self, url: str):
        """Download a file from a url."""
        # Get the file content
        response = self.session.get(self.base_url + url, allow_redirects=True)
        # Get the filename
        filename = self.parse_filename(response.url)
        # Write the file to disk
        with open(Path(self.folder_path) / filename, "wb") as f:
            f.write(response.content)

    def parse_filename(self, url: str) -> str:
        """Parse the filename from a url."""
        # Check if the file path is a pdf
        filename = Path(url)
        if filename.suffix == ".pdf":
            # If so, return the file path
            filename = filename.name
        else:
            # Using regex to get the filename from the url
            filename_matches = re.search(r"filename%2A%3DUTF-8%27%27(.+)", url)
            # If the regex matches
            if filename_matches:
                # Get the filename from the first group
                filename = filename_matches.group(1)
            else:
                # Otherwise, raise
                raise ValueError(f"Could not parse filename from {url}")
        # Check to see if it is a pdf
        if ".pdf" not in filename:
            # If not a pdf, raise
            raise ValueError(f"Incorrect file type: {filename}")
        # Remove everything after the .pdf
        filename = filename.split(".pdf")[0] + ".pdf"
        # Url decode the filename
        filename = unquote(filename)
        # Replace %20 with spaces
        filename = filename.replace("%20", " ")
        # Return the filename
        return filename

    def parse_qs(query_string: str) -> dict:
        """Parse a query string into a dict."""
        # Split the query string into a list of key=value pairs
        pairs = query_string.split("&")
        # Initialize an empty dict
        query_dict = {}
        # Iterate over the list of key=value pairs
        for pair in pairs:
            # Split the pair into a list of key and value
            key, value = pair.split("=")
            # Add the key and value to the dict
            query_dict[key] = value
        # Return the dict
        return query_dict


if __name__ == "__main__":
    loader = BlackboardLoader(
        "https://<YOUR BLACKBOARD URL HERE>/webapps/blackboard/content/listContent.jsp?course_id=_<YOUR COURSE ID HERE>_1&content_id=_<YOUR CONTENT ID HERE>_1&mode=reset",
        load_all_recursively=True,
        bbrouter="<YOUR BBROUTER COOKIE HERE>",
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} pages of PDFs from {loader.web_path}")
