import contextlib
import re
from pathlib import Path
from typing import Any, List, Optional, Tuple
from urllib.parse import unquote

from langchain.docstore.document import Document
from langchain.document_loaders.directory import DirectoryLoader
from langchain.document_loaders.pdf import PyPDFLoader
from langchain.document_loaders.web_base import WebBaseLoader


class BlackboardLoader(WebBaseLoader):
    """Load a `Blackboard` course.

    This loader is not compatible with all Blackboard courses. It is only
    compatible with courses that use the new Blackboard interface.
    To use this loader, you must have the BbRouter cookie. You can get this
    cookie by logging into the course and then copying the value of the
    BbRouter cookie from the browser's developer tools.

    Example:
        .. code-block:: python

            from langchain.document_loaders import BlackboardLoader

            loader = BlackboardLoader(
                blackboard_course_url="https://blackboard.example.com/webapps/blackboard/execute/announcement?method=search&context=course_entry&course_id=_123456_1",
                bbrouter="expires:12345...",
            )
            documents = loader.load()

    """  # noqa: E501

    def __init__(
        self,
        blackboard_course_url: str,
        bbrouter: str,
        load_all_recursively: bool = True,
        basic_auth: Optional[Tuple[str, str]] = None,
        cookies: Optional[dict] = None,
        continue_on_failure: bool = False,
    ):
        """Initialize with blackboard course url.

        The BbRouter cookie is required for most blackboard courses.

        Args:
            blackboard_course_url: Blackboard course url.
            bbrouter: BbRouter cookie.
            load_all_recursively: If True, load all documents recursively.
            basic_auth: Basic auth credentials.
            cookies: Cookies.
            continue_on_failure: whether to continue loading the sitemap if an error
                occurs loading a url, emitting a warning instead of raising an
                exception. Setting this to True makes the loader more robust, but also
                may result in missing data. Default: False

        Raises:
            ValueError: If blackboard course url is invalid.
        """
        super().__init__(
            web_paths=(blackboard_course_url), continue_on_failure=continue_on_failure
        )
        # Get base url
        try:
            self.base_url = blackboard_course_url.split("/webapps/blackboard")[0]
        except IndexError:
            raise IndexError(
                "Invalid blackboard course url. "
                "Please provide a url that starts with "
                "https://<blackboard_url>/webapps/blackboard"
            )
        if basic_auth is not None:
            self.session.auth = basic_auth
        # Combine cookies
        if cookies is None:
            cookies = {}
        cookies.update({"BbRouter": bbrouter})
        self.session.cookies.update(cookies)
        self.load_all_recursively = load_all_recursively
        self.check_bs4()

    def check_bs4(self) -> None:
        """Check if BeautifulSoup4 is installed.

        Raises:
            ImportError: If BeautifulSoup4 is not installed.
        """
        try:
            import bs4  # noqa: F401
        except ImportError:
            raise ImportError(
                "BeautifulSoup4 is required for BlackboardLoader. "
                "Please install it with `pip install beautifulsoup4`."
            )

    def load(self) -> List[Document]:
        """Load data into Document objects.

        Returns:
            List of Documents.
        """
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

    def _get_folder_path(self, soup: Any) -> str:
        """Get the folder path to save the Documents in.

        Args:
            soup: BeautifulSoup4 soup object.

        Returns:
            Folder path.
        """
        # Get the course name
        course_name = soup.find("span", {"id": "crumb_1"})
        if course_name is None:
            raise ValueError("No course name found.")
        course_name = course_name.text.strip()
        # Prepare the folder path
        course_name_clean = (
            unquote(course_name)
            .replace(" ", "_")
            .replace("/", "_")
            .replace(":", "_")
            .replace(",", "_")
            .replace("?", "_")
            .replace("'", "_")
            .replace("!", "_")
            .replace('"', "_")
        )
        # Get the folder path
        folder_path = Path(".") / course_name_clean
        return str(folder_path)

    def _get_documents(self, soup: Any) -> List[Document]:
        """Fetch content from page and return Documents.

        Args:
            soup: BeautifulSoup4 soup object.

        Returns:
            List of documents.
        """
        attachments = self._get_attachments(soup)
        self._download_attachments(attachments)
        documents = self._load_documents()
        return documents

    def _get_attachments(self, soup: Any) -> List[str]:
        """Get all attachments from a page.

        Args:
            soup: BeautifulSoup4 soup object.

        Returns:
            List of attachments.
        """
        from bs4 import BeautifulSoup, Tag

        # Get content list
        content_list = soup.find("ul", {"class": "contentList"})
        if content_list is None:
            raise ValueError("No content list found.")
        content_list: BeautifulSoup  # type: ignore
        # Get all attachments
        attachments = []
        for attachment in content_list.find_all("ul", {"class": "attachments"}):
            attachment: Tag  # type: ignore
            for link in attachment.find_all("a"):
                link: Tag  # type: ignore
                href = link.get("href")
                # Only add if href is not None and does not start with #
                if href is not None and not href.startswith("#"):
                    attachments.append(href)
        return attachments

    def _download_attachments(self, attachments: List[str]) -> None:
        """Download all attachments.

        Args:
            attachments: List of attachments.
        """
        # Make sure the folder exists
        Path(self.folder_path).mkdir(parents=True, exist_ok=True)
        # Download all attachments
        for attachment in attachments:
            self.download(attachment)

    def _load_documents(self) -> List[Document]:
        """Load all documents in the folder.

        Returns:
            List of documents.
        """
        # Create the document loader
        loader = DirectoryLoader(
            path=self.folder_path, glob="*.pdf", loader_cls=PyPDFLoader  # type: ignore
        )
        # Load the documents
        documents = loader.load()
        # Return all documents
        return documents

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

    def download(self, path: str) -> None:
        """Download a file from an url.

        Args:
            path: Path to the file.
        """
        # Get the file content
        response = self.session.get(self.base_url + path, allow_redirects=True)
        # Get the filename
        filename = self.parse_filename(response.url)
        # Write the file to disk
        with open(Path(self.folder_path) / filename, "wb") as f:
            f.write(response.content)

    def parse_filename(self, url: str) -> str:
        """Parse the filename from an url.

        Args:
            url: Url to parse the filename from.

        Returns:
            The filename.
        """
        if (url_path := Path(url)) and url_path.suffix == ".pdf":
            return url_path.name
        else:
            return self._parse_filename_from_url(url)

    def _parse_filename_from_url(self, url: str) -> str:
        """Parse the filename from an url.

        Args:
            url: Url to parse the filename from.

        Returns:
            The filename.

        Raises:
            ValueError: If the filename could not be parsed.
        """
        filename_matches = re.search(r"filename%2A%3DUTF-8%27%27(.+)", url)
        if filename_matches:
            filename = filename_matches.group(1)
        else:
            raise ValueError(f"Could not parse filename from {url}")
        if ".pdf" not in filename:
            raise ValueError(f"Incorrect file type: {filename}")
        filename = filename.split(".pdf")[0] + ".pdf"
        filename = unquote(filename)
        filename = filename.replace("%20", " ")
        return filename


if __name__ == "__main__":
    loader = BlackboardLoader(
        "https://<YOUR BLACKBOARD URL"
        " HERE>/webapps/blackboard/content/listContent.jsp?course_id=_<YOUR COURSE ID"
        " HERE>_1&content_id=_<YOUR CONTENT ID HERE>_1&mode=reset",
        "<YOUR BBROUTER COOKIE HERE>",
        load_all_recursively=True,
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} pages of PDFs from {loader.web_path}")
