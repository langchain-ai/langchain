from typing import Any, List, Tuple

from langchain.chains import MapReduceDocumentsChain, StuffDocumentsChain
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.pydantic_v1 import Extra
from langchain.schema import Document


class ConditionalDocumentsChain(BaseCombineDocumentsChain):
    """
    Цепь, которая выполняет StuffDocumentsChain,
    если размер документов меньше max_length,
    если же размер больше, то выполняет MapReduceDocumentsChain
    """

    stuff_chain: StuffDocumentsChain
    map_reduce_chain: MapReduceDocumentsChain
    # Размер указывается в токенах
    max_length: int = 2000

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    def combine_docs(self, docs: List[Document], **kwargs: Any) -> Tuple[str, dict]:
        current_chain = self.current_chain(docs, **kwargs)
        return current_chain.combine_docs(docs, **kwargs)

    async def acombine_docs(
        self, docs: List[Document], **kwargs: Any
    ) -> Tuple[str, dict]:
        current_chain = self.current_chain(docs, **kwargs)
        return await current_chain.acombine_docs(docs, **kwargs)

    @property
    def _chain_type(self) -> str:
        return "conditional_documents_chain"

    def current_chain(
        self, docs: List[Document], **kwargs: Any
    ) -> BaseCombineDocumentsChain:
        prompt_length = self.stuff_chain.prompt_length(docs, **kwargs)
        if prompt_length is None or prompt_length > self.max_length:
            return self.map_reduce_chain
        return self.stuff_chain
