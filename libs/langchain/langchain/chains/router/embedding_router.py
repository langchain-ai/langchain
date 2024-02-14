from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Type

from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import Extra
from langchain_core.vectorstores import VectorStore

from langchain.chains.router.base import RouterChain


class EmbeddingRouterChain(RouterChain):
    """Chain that uses embeddings to route between options."""

    vectorstore: VectorStore
    routing_keys: List[str] = ["query"]

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Will be whatever keys the LLM chain prompt expects.

        :meta private:
        """
        return self.routing_keys

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        _input = ", ".join([inputs[k] for k in self.routing_keys])
        results = self.vectorstore.similarity_search(_input, k=1)
        return {"next_inputs": inputs, "destination": results[0].metadata["name"]}

    @classmethod
    def from_names_and_descriptions(
        cls,
        names_and_descriptions: Sequence[Tuple[str, Sequence[str]]],
        vectorstore_cls: Type[VectorStore],
        embeddings: Embeddings,
        **kwargs: Any,
    ) -> EmbeddingRouterChain:
        """Convenience constructor."""
        documents = []
        for name, descriptions in names_and_descriptions:
            for description in descriptions:
                documents.append(
                    Document(page_content=description, metadata={"name": name})
                )
        vectorstore = vectorstore_cls.from_documents(documents, embeddings)
        return cls(vectorstore=vectorstore, **kwargs)
