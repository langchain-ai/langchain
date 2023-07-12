from typing import Any, Optional, Sequence

from langchain.schema import BaseDocumentTransformer, Document
from langchain.utils import get_from_env


class DoctranQATranslater(BaseDocumentTransformer):
    """Translates text documents using doctran."""

    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        raise NotImplementedError

    async def atransform_documents(
        self,
        documents: Sequence[Document],
        language: str,
        openai_api_key: Optional[str] = None,
        **kwargs: Any
    ) -> Sequence[Document]:
        """Translates text documents using doctran."""

        if openai_api_key:
            openai_api_key = openai_api_key
        else:
            openai_api_key = get_from_env("openai_api_key", "OPENAI_API_KEY")
        try:
            from doctran import Doctran

            doctran = Doctran(openai_api_key=openai_api_key)
        except ImportError:
            raise ImportError(
                "Install doctran to use this parser. (pip install doctran)"
            )
        for d in documents:
            transformed_content = (
                doctran.parse(content=d.page_content)
                .translate(language=language)
                .execute()
            )
            d.metadata["translation"] = transformed_content
        return documents
