from typing import Any, Optional, Sequence

from langchain.schema import BaseDocumentTransformer, Document
from langchain.utils import get_from_env


class DoctranQATransformer(BaseDocumentTransformer):
    """Extracts QA from text documents using doctran."""

    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        raise NotImplementedError

    async def atransform_documents(
        self,
        documents: Sequence[Document],
        openai_api_key: Optional[str] = None,
        **kwargs: Any
    ) -> Sequence[Document]:
        """Extracts QA from text documents using doctran."""

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
            doctran_doc = await doctran.parse(content=d.page_content).interrogate().execute()
            questions_and_answers = doctran_doc.extracted_properties.get(
                "questions_and_answers"
            )
            d.metadata["questions_and_answers"] = questions_and_answers
        return documents
