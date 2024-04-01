from typing import Any, List, Optional, Sequence

from langchain_core.documents import BaseDocumentTransformer, Document
from langchain_core.utils import get_from_env


class DoctranPropertyExtractor(BaseDocumentTransformer):
    """Extract properties from text documents using doctran.

    Arguments:
        properties: A list of the properties to extract.
        openai_api_key: OpenAI API key. Can also be specified via environment variable
            ``OPENAI_API_KEY``.

    Example:
        .. code-block:: python

            from langchain_community.document_transformers import DoctranPropertyExtractor

            properties = [
                {
                    "name": "category",
                    "description": "What type of email this is.",
                    "type": "string",
                    "enum": ["update", "action_item", "customer_feedback", "announcement", "other"],
                    "required": True,
                },
                {
                    "name": "mentions",
                    "description": "A list of all people mentioned in this email.",
                    "type": "array",
                    "items": {
                        "name": "full_name",
                        "description": "The full name of the person mentioned.",
                        "type": "string",
                    },
                    "required": True,
                },
                {
                    "name": "eli5",
                    "description": "Explain this email to me like I'm 5 years old.",
                    "type": "string",
                    "required": True,
                },
            ]

            # Pass in openai_api_key or set env var OPENAI_API_KEY
            property_extractor = DoctranPropertyExtractor(properties)
            transformed_document = await qa_transformer.atransform_documents(documents)
    """  # noqa: E501

    def __init__(
        self,
        properties: List[dict],
        openai_api_key: Optional[str] = None,
        openai_api_model: Optional[str] = None,
    ) -> None:
        self.properties = properties
        self.openai_api_key = openai_api_key or get_from_env(
            "openai_api_key", "OPENAI_API_KEY"
        )
        self.openai_api_model = openai_api_model or get_from_env(
            "openai_api_model", "OPENAI_API_MODEL"
        )

    async def atransform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        """Extracts properties from text documents using doctran."""
        try:
            from doctran import Doctran, ExtractProperty

            doctran = Doctran(
                openai_api_key=self.openai_api_key, openai_model=self.openai_api_model
            )
        except ImportError:
            raise ImportError(
                "Install doctran to use this parser. (pip install doctran)"
            )
        properties = [ExtractProperty(**property) for property in self.properties]
        for d in documents:
            doctran_doc = (
                doctran.parse(content=d.page_content)
                .extract(properties=properties)
                .execute()
            )

            d.metadata["extracted_properties"] = doctran_doc.extracted_properties
        return documents

    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        """Extracts properties from text documents using doctran."""
        try:
            from doctran import Doctran, ExtractProperty

            doctran = Doctran(
                openai_api_key=self.openai_api_key, openai_model=self.openai_api_model
            )
        except ImportError:
            raise ImportError(
                "Install doctran to use this parser. (pip install doctran)"
            )
        properties = [ExtractProperty(**property) for property in self.properties]
        for d in documents:
            doctran_doc = (
                doctran.parse(content=d.page_content)
                .extract(properties=properties)
                .execute()
            )

            d.metadata["extracted_properties"] = doctran_doc.extracted_properties
        return documents
