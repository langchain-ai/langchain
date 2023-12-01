import asyncio
import json
import uuid
from typing import Any, Sequence

from langchain_core.documents import BaseDocumentTransformer, Document

from langchain.tools.nuclia.tool import NucliaUnderstandingAPI


class NucliaTextTransformer(BaseDocumentTransformer):
    """
    The Nuclia Understanding API splits into paragraphs and sentences,
    identifies entities, provides a summary of the text and generates
    embeddings for all sentences.
    """

    def __init__(self, nua: NucliaUnderstandingAPI):
        self.nua = nua

    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        raise NotImplementedError

    async def atransform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        tasks = [
            self.nua.arun(
                {
                    "action": "push",
                    "id": str(uuid.uuid4()),
                    "text": doc.page_content,
                    "path": None,
                }
            )
            for doc in documents
        ]
        results = await asyncio.gather(*tasks)
        for doc, result in zip(documents, results):
            obj = json.loads(result)
            metadata = {
                "file": obj["file_extracted_data"][0],
                "metadata": obj["field_metadata"][0],
            }
            doc.metadata["nuclia"] = metadata
        return documents
