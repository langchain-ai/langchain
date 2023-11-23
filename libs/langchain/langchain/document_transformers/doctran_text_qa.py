from typing import Any, Optional, Sequence

from langchain_core.documents import BaseDocumentTransformer, Document

from langchain.utils import get_from_env


class DoctranQATransformer(BaseDocumentTransformer):
    """Extract QA from text documents using doctran.

    Arguments:
        openai_api_key: OpenAI API key. Can also be specified via environment variable
            ``OPENAI_API_KEY``.

    Example:
        .. code-block:: python

            from langchain.document_transformers import DoctranQATransformer

            # Pass in openai_api_key or set env var OPENAI_API_KEY
            qa_transformer = DoctranQATransformer()
            transformed_document = await qa_transformer.atransform_documents(documents)
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        openai_api_model: Optional[str] = None,
    ) -> None:
        self.openai_api_key = openai_api_key or get_from_env(
            "openai_api_key", "OPENAI_API_KEY"
        )
        self.openai_api_model = openai_api_model or get_from_env(
            "openai_api_model", "OPENAI_API_MODEL"
        )

    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        raise NotImplementedError

    async def atransform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        """Extracts QA from text documents using doctran."""
        try:
            from doctran import Doctran

            doctran = Doctran(
                openai_api_key=self.openai_api_key, openai_model=self.openai_api_model
            )
        except ImportError:
            raise ImportError(
                "Install doctran to use this parser. (pip install doctran)"
            )
        for d in documents:
            doctran_doc = (
                await doctran.parse(content=d.page_content).interrogate().execute()
            )
            questions_and_answers = doctran_doc.extracted_properties.get(
                "questions_and_answers"
            )
            d.metadata["questions_and_answers"] = questions_and_answers
        return documents
