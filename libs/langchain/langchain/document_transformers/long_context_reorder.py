"""Reorder documents"""
from typing import Any, List, Sequence

from langchain.pydantic_v1 import BaseModel
from langchain.schema import BaseDocumentTransformer, Document


def _litm_reordering(documents: List[Document]) -> List[Document]:
    """Los in the middle reorder: the most relevant will be at the
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
    """Lost in the middle:
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
        raise NotImplementedError
