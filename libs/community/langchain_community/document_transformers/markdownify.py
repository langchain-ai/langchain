import re
from typing import Any, List, Optional, Sequence, Union

from langchain_core.documents import BaseDocumentTransformer, Document


class MarkdownifyTransformer(BaseDocumentTransformer):
    """Converts HTML documents to Markdown format with customizable options for handling
    links, images, other tags and heading styles using the markdownify library.

    Arguments:
        strip: A list of tags to strip. This option can't be used with the convert option.
        convert: A list of tags to convert. This option can't be used with the strip option.
        autolinks: A boolean indicating whether the "automatic link" style should be used when a a tag's contents match its href. Defaults to True.
        heading_style: Defines how headings should be converted. Accepted values are ATX, ATX_CLOSED, SETEXT, and UNDERLINED (which is an alias for SETEXT). Defaults to ATX.
        kwargs: Additional options to pass to markdownify.

    Example:
        .. code-block:: python
            from langchain_community.document_transformers import MarkdownifyTransformer
            markdownify = MarkdownifyTransformer()
            docs_transform = markdownify.transform_documents(docs)

    More configuration options can be found at the markdownify GitHub page:
    https://github.com/matthewwithanm/python-markdownify
    """  # noqa: E501

    def __init__(
        self,
        strip: Optional[Union[str, List[str]]] = None,
        convert: Optional[Union[str, List[str]]] = None,
        autolinks: bool = True,
        heading_style: str = "ATX",
        **kwargs: Any,
    ) -> None:
        self.strip = [strip] if isinstance(strip, str) else strip
        self.convert = [convert] if isinstance(convert, str) else convert
        self.autolinks = autolinks
        self.heading_style = heading_style
        self.additional_options = kwargs

    def transform_documents(
        self,
        documents: Sequence[Document],
        **kwargs: Any,
    ) -> Sequence[Document]:
        try:
            from markdownify import markdownify
        except ImportError:
            raise ImportError(
                """markdownify package not found, please 
                install it with `pip install markdownify`"""
            )

        converted_documents = []
        for doc in documents:
            markdown_content = (
                markdownify(
                    html=doc.page_content,
                    strip=self.strip,
                    convert=self.convert,
                    autolinks=self.autolinks,
                    heading_style=self.heading_style,
                    **self.additional_options,
                )
                .replace("\xa0", " ")
                .strip()
            )

            cleaned_markdown = re.sub(r"\n\s*\n", "\n\n", markdown_content)

            converted_documents.append(
                Document(cleaned_markdown, metadata=doc.metadata)
            )

        return converted_documents
