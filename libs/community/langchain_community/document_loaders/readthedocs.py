from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, List, Optional, Sequence, Tuple, Union

from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader

if TYPE_CHECKING:
    from bs4 import NavigableString
    from bs4.element import Comment, Tag


class ReadTheDocsLoader(BaseLoader):
    """Load `ReadTheDocs` documentation directory."""

    def __init__(
        self,
        path: Union[str, Path],
        encoding: Optional[str] = None,
        errors: Optional[str] = None,
        custom_html_tag: Optional[Tuple[str, dict]] = None,
        patterns: Sequence[str] = ("*.htm", "*.html"),
        exclude_links_ratio: float = 1.0,
        **kwargs: Optional[Any],
    ):
        """
        Initialize ReadTheDocsLoader

        The loader loops over all files under `path` and extracts the actual content of
        the files by retrieving main html tags. Default main html tags include
        `<main id="main-content>`, <`div role="main>`, and `<article role="main">`. You
        can also define your own html tags by passing custom_html_tag, e.g.
        `("div", "class=main")`. The loader iterates html tags with the order of
        custom html tags (if exists) and default html tags. If any of the tags is not
        empty, the loop will break and retrieve the content out of that tag.

        Args:
            path: The location of pulled readthedocs folder.
            encoding: The encoding with which to open the documents.
            errors: Specify how encoding and decoding errors are to be handled—this
                cannot be used in binary mode.
            custom_html_tag: Optional custom html tag to retrieve the content from
                files.
            patterns: The file patterns to load, passed to `glob.rglob`.
            exclude_links_ratio: The ratio of links:content to exclude pages from.
                This is to reduce the frequency at which index pages make their
                way into retrieved results. Recommended: 0.5
            kwargs: named arguments passed to `bs4.BeautifulSoup`.
        """
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError(
                "Could not import python packages. "
                "Please install it with `pip install beautifulsoup4`. "
            )

        try:
            _ = BeautifulSoup(
                "<html><body>Parser builder library test.</body></html>",
                "html.parser",
                **kwargs,
            )
        except Exception as e:
            raise ValueError("Parsing kwargs do not appear valid") from e

        self.file_path = Path(path)
        self.encoding = encoding
        self.errors = errors
        self.custom_html_tag = custom_html_tag
        self.patterns = patterns
        self.bs_kwargs = kwargs
        self.exclude_links_ratio = exclude_links_ratio

    def lazy_load(self) -> Iterator[Document]:
        """A lazy loader for Documents."""
        for file_pattern in self.patterns:
            for p in self.file_path.rglob(file_pattern):
                if p.is_dir():
                    continue
                with open(p, encoding=self.encoding, errors=self.errors) as f:
                    text = self._clean_data(f.read())
                yield Document(page_content=text, metadata={"source": str(p)})

    def load(self) -> List[Document]:
        """Load documents."""
        return list(self.lazy_load())

    def _clean_data(self, data: str) -> str:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(data, "html.parser", **self.bs_kwargs)

        # default tags
        html_tags = [
            ("div", {"role": "main"}),
            ("main", {"id": "main-content"}),
        ]

        if self.custom_html_tag is not None:
            html_tags.append(self.custom_html_tag)

        element = None

        # reversed order. check the custom one first
        for tag, attrs in html_tags[::-1]:
            element = soup.find(tag, attrs)
            # if found, break
            if element is not None:
                break

        if element is not None and _get_link_ratio(element) <= self.exclude_links_ratio:
            text = _get_clean_text(element)
        else:
            text = ""
        # trim empty lines
        return "\n".join([t for t in text.split("\n") if t])


def _get_clean_text(element: Tag) -> str:
    """Returns cleaned text with newlines preserved and irrelevant elements removed."""
    elements_to_skip = [
        "script",
        "noscript",
        "canvas",
        "meta",
        "svg",
        "map",
        "area",
        "audio",
        "source",
        "track",
        "video",
        "embed",
        "object",
        "param",
        "picture",
        "iframe",
        "frame",
        "frameset",
        "noframes",
        "applet",
        "form",
        "button",
        "select",
        "base",
        "style",
        "img",
    ]

    newline_elements = [
        "p",
        "div",
        "ul",
        "ol",
        "li",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "pre",
        "table",
        "tr",
    ]

    text = _process_element(element, elements_to_skip, newline_elements)
    return text.strip()


def _get_link_ratio(section: Tag) -> float:
    links = section.find_all("a")
    total_text = "".join(str(s) for s in section.stripped_strings)
    if len(total_text) == 0:
        return 0

    link_text = "".join(
        str(string.string.strip())
        for link in links
        for string in link.strings
        if string
    )
    return len(link_text) / len(total_text)


def _process_element(
    element: Union[Tag, NavigableString, Comment],
    elements_to_skip: List[str],
    newline_elements: List[str],
) -> str:
    """
    Traverse through HTML tree recursively to preserve newline and skip
    unwanted (code/binary) elements
    """
    from bs4 import NavigableString
    from bs4.element import Comment, Tag

    tag_name = getattr(element, "name", None)
    if isinstance(element, Comment) or tag_name in elements_to_skip:
        return ""
    elif isinstance(element, NavigableString):
        return element
    elif tag_name == "br":
        return "\n"
    elif tag_name in newline_elements:
        return (
            "".join(
                _process_element(child, elements_to_skip, newline_elements)
                for child in element.children
                if isinstance(child, (Tag, NavigableString, Comment))
            )
            + "\n"
        )
    else:
        return "".join(
            _process_element(child, elements_to_skip, newline_elements)
            for child in element.children
            if isinstance(child, (Tag, NavigableString, Comment))
        )
