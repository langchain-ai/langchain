from typing import Any, Sequence

from langchain.schema import BaseDocumentTransformer, Document
from langchain.utils import get_from_dict_or_env


class DoctranTextTranslator(BaseDocumentTransformer):
    """Translates text documents using doctran."""

    def __init__(self, **kwargs: Any) -> None:
        self.openai_api_key = get_from_dict_or_env(
            kwargs, "openai_api_key", "OPENAI_API_KEY"
        )
        self.language = kwargs.get("language", "english")

    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        raise NotImplementedError

    async def atransform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        """Translates text documents using doctran."""
        try:
            from doctran import Doctran

            doctran = Doctran(openai_api_key=self.openai_api_key)
        except ImportError:
            raise ImportError(
                "Install doctran to use this parser. (pip install doctran)"
            )
        doctran_docs = [
            doctran.parse(content=doc.page_content, metadata=doc.metadata)
            for doc in documents
        ]
        for i, doc in enumerate(doctran_docs):
            doctran_docs[i] = await doc.translate(language=self.language).execute()
        return [
            Document(page_content=doc.transformed_content, metadata=doc.metadata)
            for doc in doctran_docs
        ]
