from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Sequence

from langchain.callbacks.manager import Callbacks
from langchain.pydantic_v1 import Extra, root_validator
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain.schema import Document
from langchain.utils import get_from_dict_or_env

if TYPE_CHECKING:
    from cohere import Client
else:
    # We do to avoid pydantic annotation issues when actually instantiating
    # while keeping this import optional
    try:
        from cohere import Client
    except ImportError:
        pass


class CohereRerank(BaseDocumentCompressor):
    """Document compressor that uses `Cohere Rerank API`."""

    client: Client
    """Cohere client to use for compressing documents."""
    top_n: int = 3
    """Number of documents to return."""
    model: str = "rerank-english-v2.0"
    """Model to use for reranking."""

    cohere_api_key: Optional[str] = None
    user_agent: str = "langchain"
    """Identifier for the application making the request."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        try:
            import cohere
        except ImportError:
            raise ImportError(
                "Could not import cohere python package. "
                "Please install it with `pip install cohere`."
            )
        cohere_api_key = get_from_dict_or_env(
            values, "cohere_api_key", "COHERE_API_KEY"
        )
        client_name = values["user_agent"]
        values["client"] = cohere.Client(cohere_api_key, client_name=client_name)
        return values

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """
        Compress documents using Cohere's rerank API.

        Args:
            documents: A sequence of documents to compress.
            query: The query to use for compressing the documents.
            callbacks: Callbacks to run during the compression process.

        Returns:
            A sequence of compressed documents.
        """
        if len(documents) == 0:  # to avoid empty api call
            return []
        doc_list = list(documents)
        _docs = [d.page_content for d in doc_list]
        results = self.client.rerank(
            model=self.model, query=query, documents=_docs, top_n=self.top_n
        )
        final_results = []
        for r in results:
            doc = doc_list[r.index]
            doc.metadata["relevance_score"] = r.relevance_score
            final_results.append(doc)
        return final_results
