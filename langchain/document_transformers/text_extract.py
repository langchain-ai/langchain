from typing import Any, List, Optional, Sequence

from langchain.schema import BaseDocumentTransformer, Document
from langchain.utils import get_from_env


class DoctranPropertyExtractor(BaseDocumentTransformer):
    """Extracts properties from text documents using doctran."""

    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        raise NotImplementedError

    async def atransform_documents(
        self,
        documents: Sequence[Document],
        properties: List[dict],
        openai_api_key: Optional[str] = None,
        **kwargs: Any
    ) -> Sequence[Document]:
        """Extracts properties from text documents using doctran."""

        if openai_api_key:
            openai_api_key = openai_api_key
        else:
            openai_api_key = get_from_env("openai_api_key", "OPENAI_API_KEY")
        try:
            from doctran import Doctran, ExtractProperty

            doctran = Doctran(openai_api_key=openai_api_key)
        except ImportError:
            raise ImportError(
                "Install doctran to use this parser. (pip install doctran)"
            )
        properties = [ExtractProperty(**property) for property in properties]
        for d in documents:
            doctran_doc = await (
                doctran.parse(content=d.page_content)
                .extract(properties=properties)
                .execute()
            )
            d.metadata["extracted_properties"] = doctran_doc.extracted_properties
        return documents
