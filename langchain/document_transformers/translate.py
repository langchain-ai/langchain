"""Transform documents"""
from typing import Any, Sequence

from pydantic import BaseModel

from langchain.utils import get_from_dict_or_env
from langchain.schema import BaseDocumentTransformer, Document
from doctran import Doctran


class DocumentTranslator(BaseDocumentTransformer, BaseModel):
    """Translates documents into another language."""

    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        raise NotImplementedError


    async def atransform_documents(
        self, documents: Sequence[Document], language: str, **kwargs: Any
    ) -> Sequence[Document]:
        """Asynchronously translate documents into another language doctran and OpenAI functions."""
        openai_api_key = get_from_dict_or_env(kwargs, "openai_api_key", "OPENAI_API_KEY")
        doctran = Doctran(openai_api_key=openai_api_key)
        doctran_docs = [doctran.parse(content=doc.page_content, metadata=doc.metadata) for doc in documents]
        for i, doc in enumerate(doctran_docs):
            try:
                doctran_docs[i] = await doc.translate(language=language).execute()
            except Exception as e:
                pass
        return [Document(page_content=doc.transformed_content) for doc in doctran_docs]