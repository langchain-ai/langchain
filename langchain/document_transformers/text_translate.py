from typing import Any, Sequence

from langchain.schema import BaseDocumentTransformer, Document
from langchain.utils import get_from_env


class DoctranTextTranslator(BaseDocumentTransformer):
    """Translates text documents using doctran."""

    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        raise NotImplementedError

    async def atransform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        """Translates text documents using doctran."""

        language = kwargs.get("language", "english")
        openai_api_key = kwargs.get("openai_api_key", None)
        if not openai_api_key:
            openai_api_key = get_from_env("openai_api_key", "OPENAI_API_KEY")

        try:
            from doctran import Doctran

            doctran = Doctran(openai_api_key=openai_api_key)
        except ImportError:
            raise ImportError(
                "Install doctran to use this parser. (pip install doctran)"
            )
        doctran_docs = [
            doctran.parse(content=doc.page_content, metadata=doc.metadata)
            for doc in documents
        ]
        for i, doc in enumerate(doctran_docs):
            doctran_docs[i] = await doc.translate(language=language).execute()
        return [Document(page_content=doc.transformed_content) for doc in doctran_docs]
