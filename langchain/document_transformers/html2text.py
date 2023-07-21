from typing import Any, Sequence

from langchain.schema import BaseDocumentTransformer, Document


class Html2TextTransformer(BaseDocumentTransformer):
    """Replace occurrences of a particular search pattern with a replacement string
    Example:
        .. code-block:: python
            from langchain.document_transformers import Html2TextTransformer
            html2text=Html2TextTransformer()
            docs_transform=html2text.transform_documents(docs)
    """

    def transform_documents(
        self,
        documents: Sequence[Document],
        **kwargs: Any,
    ) -> Sequence[Document]:
        try:
            import html2text
        except ImportError:
            raise ValueError(
                """html2text package not found, please 
                install it with `pip install html2text`"""
            )

        # Create an html2text.HTML2Text object and override some properties
        h = html2text.HTML2Text()
        h.ignore_links = True
        h.ignore_images = True
        for d in documents:
            d.page_content = h.handle(d.page_content)
        return documents

    async def atransform_documents(
        self,
        documents: Sequence[Document],
        **kwargs: Any,
    ) -> Sequence[Document]:
        raise NotImplementedError
