from typing import Any, Sequence

from langchain.schema import BaseDocumentTransformer, Document
from langchain.utils import get_from_dict_or_env


class DoctranPropertyExtractor(BaseDocumentTransformer):
    """Extracts properties from text documents using doctran."""

    def __init__(self, **kwargs: Any) -> None:
        self.openai_api_key = get_from_dict_or_env(
            kwargs, "openai_api_key", "OPENAI_API_KEY"
        )
        self.properties = kwargs.get("properties", None)

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
