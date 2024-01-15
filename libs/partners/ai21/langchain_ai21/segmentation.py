from typing import Sequence, Any

from langchain_ai21.ai21_base import AI21Base
from langchain_core.documents import BaseDocumentTransformer, Document


class AI21Segmentation(BaseDocumentTransformer, AI21Base):
    def transform_documents(self, documents: Sequence[Document], **kwargs: Any) -> Sequence[Document]:
        new_documents = []

        for document in documents:
            segments = self.client.segmentation.create(source=document.page_content)
            for segment in segments.segments:
                new_documents.append(Document(page_content=segment, metadata={"segment_type": segment.segment_type}))

        return new_documents
