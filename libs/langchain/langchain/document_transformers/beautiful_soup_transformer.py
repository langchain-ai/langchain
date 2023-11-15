from typing import Any, Iterator, List, Sequence, cast

from langchain.schema import BaseDocumentTransformer, Document


class BeautifulSoupTransformer(BaseDocumentTransformer):
    """Transform HTML content by extracting specific tags and removing unwanted ones.

    Example:
        .. code-block:: python
            from langchain.document_transformers import BeautifulSoupTransformer
            bs4_transformer = BeautifulSoupTransformer()
            docs_transformed = bs4_transformer.transform_documents(docs)
    """

    def __init__(self) -> None:
        """
        Initialize the transformer.

        This checks if the BeautifulSoup4 package is installed.
        If not, it raises an ImportError.
        """
        try:
            import bs4  # noqa:F401
        except ImportError:
            raise ImportError(
                "BeautifulSoup4 is required for BeautifulSoupTransformer. "
                "Please install it with `pip install beautifulsoup4`."
            )

    def transform_documents(
        self,
        documents: Sequence[Document],
        unwanted_tags: List[str] = ["script", "style"],
        tags_to_extract: List[str] = ["p", "li", "div", "a"],
        remove_lines: bool = True,
        **kwargs: Any,
    ) -> Sequence[Document]:
        """
        Transform a list of Document objects by cleaning their HTML content.

        Args:
            documents: A sequence of Document objects containing HTML content.
            unwanted_tags: A list of tags to be removed from the HTML.
            tags_to_extract: A list of tags whose content will be extracted.
            remove_lines: If set to True, unnecessary lines will be
            removed from the HTML content.

        Returns:
            A sequence of Document objects with transformed content.
        """
        for doc in documents:
            cleaned_content = doc.page_content

            cleaned_content = self.remove_unwanted_tags(cleaned_content, unwanted_tags)

            cleaned_content = self.extract_tags(cleaned_content, tags_to_extract)

            if remove_lines:
                cleaned_content = self.remove_unnecessary_lines(cleaned_content)

            doc.page_content = cleaned_content

        return documents

    @staticmethod
    def remove_unwanted_tags(html_content: str, unwanted_tags: List[str]) -> str:
        """
        Remove unwanted tags from a given HTML content.

        Args:
            html_content: The original HTML content string.
            unwanted_tags: A list of tags to be removed from the HTML.

        Returns:
            A cleaned HTML string with unwanted tags removed.
        """
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html_content, "html.parser")
        for tag in unwanted_tags:
            for element in soup.find_all(tag):
                element.decompose()
        return str(soup)

    @staticmethod
    def extract_tags(html_content: str, tags: List[str]) -> str:
        """
        Extract specific tags from a given HTML content.

        Args:
            html_content: The original HTML content string.
            tags: A list of tags to be extracted from the HTML.

        Returns:
            A string combining the content of the extracted tags.
        """
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html_content, "html.parser")
        text_parts: List[str] = []
        for element in soup.find_all():
            if element.name in tags:
                # Extract all navigable strings recursively from this element.
                text_parts += get_navigable_strings(element)

                # To avoid duplicate text, remove all descendants from the soup.
                element.decompose()

        return " ".join(text_parts)

    @staticmethod
    def remove_unnecessary_lines(content: str) -> str:
        """
        Clean up the content by removing unnecessary lines.

        Args:
            content: A string, which may contain unnecessary lines or spaces.

        Returns:
            A cleaned string with unnecessary lines removed.
        """
        lines = content.split("\n")
        stripped_lines = [line.strip() for line in lines]
        non_empty_lines = [line for line in stripped_lines if line]
        cleaned_content = " ".join(non_empty_lines)
        return cleaned_content

    async def atransform_documents(
        self,
        documents: Sequence[Document],
        **kwargs: Any,
    ) -> Sequence[Document]:
        raise NotImplementedError


def get_navigable_strings(element: Any) -> Iterator[str]:
    from bs4 import NavigableString, Tag

    for child in cast(Tag, element).children:
        if isinstance(child, Tag):
            yield from get_navigable_strings(child)
        elif isinstance(child, NavigableString):
            if (element.name == "a") and (href := element.get("href")):
                yield f"{child.strip()} ({href})"
            else:
                yield child.strip()
