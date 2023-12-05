from typing import Any, Iterator, List, Sequence, cast

from langchain_core.documents import BaseDocumentTransformer, Document


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
    @staticmethod 
    def clean_html_retain_header(self, page_content):
        """transform html to text change all h1, h2, h3 to title
        args:
            page_content: html content
        return:
            clean_text: text content"""
        from bs4 import BeautifulSoup,NavigableString
        import re
        soup = BeautifulSoup(page_content, "html.parser")

        def header_level(tag_name):
            levels = {"h1": "title ", "h2": "title ", "h3": "title "}
            return levels.get(tag_name, "")

        # Remove all script and style elements
        for script_or_style in soup(["script", "style"]):
            script_or_style.extract()

        # Iterate through document headers
        headers = soup.find_all(['h1', 'h2', 'h3'])
        text_chunks = []

        for header in headers:
            level = header_level(header.name)
            header_text = header.get_text(strip=True)

            if level and header_text:
                section_text = [f"{level}{header_text}"]
                sibling = header.next_sibling

                # Go through siblings until the next header or None
                while sibling and sibling.name not in ['h1', 'h2', 'h3', None]:
                    if isinstance(sibling, NavigableString):
                        sibling_text = sibling.strip()
                        if sibling_text:  # Skip empty strings
                            section_text.append(sibling_text)
                    elif sibling.name in ["p", "div"]:
                        sibling_text = sibling.get_text(" ", strip=True)
                        if sibling_text:  # Skip empty paragraphs or divs
                            section_text.append(sibling_text)
                    sibling = sibling.next_sibling

                # Clean and concatenate the section text, separated by a single newline
                section_cleaned = '\n'.join(filter(None, section_text))
                text_chunks.append(section_cleaned)

        # Join all text chunks with two line breaks between each section
        clean_text = "\n\n".join(text_chunks).strip()

        # Normalize whitespace by collapsing multiple spaces into one, except for newlines
        clean_text = re.sub(r'[^\S\n]+', ' ', clean_text)
        # Collapse multiple newlines into a single newline
        clean_text = re.sub(r'\n+', '\n', clean_text)

        return clean_text
    def transform_documents(
        self,
        documents: Sequence[Document],
        unwanted_tags: List[str] = ["script", "style"],
        tags_to_extract: List[str] = ["p", "li", "div", "a"],
        remove_lines: bool = True,
        retain_text_only: bool = False,
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

            if retain_text_only == False:
                cleaned_content = self.remove_unwanted_tags(cleaned_content, unwanted_tags)

                cleaned_content = self.extract_tags(cleaned_content, tags_to_extract)

                if remove_lines:
                    cleaned_content = self.remove_unnecessary_lines(cleaned_content)
            elif retain_text_only == True:
                cleaned_content = self.clean_html_retain_header(self, cleaned_content)
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
