from typing import Any, Sequence

from langchain_core.documents import BaseDocumentTransformer, Document


class Html2TextTransformer(BaseDocumentTransformer):
    """Replace occurrences of a particular search pattern with a replacement string

    Arguments:
        ignore_links: Whether links should be ignored; defaults to True.
        ignore_images: Whether images should be ignored; defaults to True.

    Example:
        .. code-block:: python
            from langchain.document_transformers import Html2TextTransformer
            html2text = Html2TextTransformer()
            docs_transform = html2text.transform_documents(docs)
    """

    def __init__(self, ignore_links: bool = True, ignore_images: bool = True) -> None:
        self.ignore_links = ignore_links
        self.ignore_images = ignore_images

    def transform_documents(
        self,
        documents: Sequence[Document],
        **kwargs: Any,
    ) -> Sequence[Document]:
        try:
            import html2text
        except ImportError:
            raise ImportError(
                """html2text package not found, please 
                install it with `pip install html2text`"""
            )

        # Create a html2text.HTML2Text object and override some properties
        h = html2text.HTML2Text()
        h.ignore_links = self.ignore_links
        h.ignore_images = self.ignore_images

        for d in documents:
            d.page_content = h.handle(d.page_content)
        return documents

    async def atransform_documents(
        self,
        documents: Sequence[Document],
        **kwargs: Any,
    ) -> Sequence[Document]:
        raise NotImplementedError
