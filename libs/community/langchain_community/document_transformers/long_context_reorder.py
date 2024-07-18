"""Reorder documents"""

from typing import Any, List, Sequence

from langchain_core.documents import BaseDocumentTransformer, Document
from langchain_core.pydantic_v1 import BaseModel


def _litm_reordering(documents: List[Document]) -> List[Document]:
    """Lost in the middle reorder: the less relevant documents will be at the
    middle of the list and more relevant elements at beginning / end.
    See: https://arxiv.org/abs//2307.03172"""

    documents.reverse()
    reordered_result = []
    for i, value in enumerate(documents):
        if i % 2 == 1:
            reordered_result.append(value)
        else:
            reordered_result.insert(0, value)
    return reordered_result


class LongContextReorder(BaseDocumentTransformer, BaseModel):
    """Reorder long context.

    Lost in the middle:
    Performance degrades when models must access relevant information
    in the middle of long contexts.
    See: https://arxiv.org/abs//2307.03172"""

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        """Reorders documents."""
        return _litm_reordering(list(documents))

    async def atransform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        return _litm_reordering(list(documents))
