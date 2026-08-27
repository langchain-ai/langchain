from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from pydantic import ConfigDict
from typing_extensions import override

from langchain_classic.chains.router.base import RouterChain


class EmbeddingRouterChain(RouterChain):
    """Chain that uses embeddings to route between options."""

    vectorstore: VectorStore
    routing_keys: list[str] = ["query"]

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    @property
    def input_keys(self) -> list[str]:
        """Will be whatever keys the LLM chain prompt expects."""
        return self.routing_keys

    @override
    def _call(
        self,
        inputs: dict[str, Any],
        run_manager: CallbackManagerForChainRun | None = None,
    ) -> dict[str, Any]:
        _input = ", ".join([inputs[k] for k in self.routing_keys])
        results = self.vectorstore.similarity_search(_input, k=1)
        return {"next_inputs": inputs, "destination": results[0].metadata["name"]}

    @override
    async def _acall(
        self,
        inputs: dict[str, Any],
        run_manager: AsyncCallbackManagerForChainRun | None = None,
    ) -> dict[str, Any]:
        _input = ", ".join([inputs[k] for k in self.routing_keys])
        results = await self.vectorstore.asimilarity_search(_input, k=1)
        return {"next_inputs": inputs, "destination": results[0].metadata["name"]}

    @classmethod
    def from_names_and_descriptions(
        cls,
        names_and_descriptions: Sequence[tuple[str, Sequence[str]]],
        vectorstore_cls: type[VectorStore],
        embeddings: Embeddings,
        **kwargs: Any,
    ) -> EmbeddingRouterChain:
        """Convenience constructor."""
        documents = []
        for name, descriptions in names_and_descriptions:
            documents.extend(
                [
                    Document(page_content=description, metadata={"name": name})
                    for description in descriptions
                ]
            )
        vectorstore = vectorstore_cls.from_documents(documents, embeddings)
        return cls(vectorstore=vectorstore, **kwargs)

    @classmethod
    async def afrom_names_and_descriptions(
        cls,
        names_and_descriptions: Sequence[tuple[str, Sequence[str]]],
        vectorstore_cls: type[VectorStore],
        embeddings: Embeddings,
        **kwargs: Any,
    ) -> EmbeddingRouterChain:
        """Convenience constructor."""
        documents = []
        documents.extend(
            [
                Document(page_content=description, metadata={"name": name})
                for name, descriptions in names_and_descriptions
                for description in descriptions
            ]
        )
        vectorstore = await vectorstore_cls.afrom_documents(documents, embeddings)
        return cls(vectorstore=vectorstore, **kwargs)
