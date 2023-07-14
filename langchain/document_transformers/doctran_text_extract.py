from typing import Any, List, Optional, Sequence

from langchain.schema import BaseDocumentTransformer, Document
from langchain.utils import get_from_env


class DoctranPropertyExtractor(BaseDocumentTransformer):
    """Extracts properties from text documents using doctran.

    Arguments:
        properties: A list of the properties to extract.
        openai_api_key: OpenAI API key. Can also be specified via environment variable
            ``OPENAI_API_KEY``.

    Example:
        .. code-block:: python

            from langchain.document_transformers import DoctranPropertyExtractor

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
    ) -> None:
        self.properties = properties
        self.openai_api_key = openai_api_key or get_from_env(
            "openai_api_key", "OPENAI_API_KEY"
        )

    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        raise NotImplementedError

    async def atransform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        """Extracts properties from text documents using doctran."""
        try:
            from doctran import Doctran, ExtractProperty

            doctran = Doctran(openai_api_key=self.openai_api_key)
        except ImportError:
            raise ImportError(
                "Install doctran to use this parser. (pip install doctran)"
            )
        properties = [ExtractProperty(**property) for property in self.properties]
        for d in documents:
            doctran_doc = (
                await doctran.parse(content=d.page_content)
                .extract(properties=properties)
                .execute()
            )

            d.metadata["extracted_properties"] = doctran_doc.extracted_properties
        return documents
