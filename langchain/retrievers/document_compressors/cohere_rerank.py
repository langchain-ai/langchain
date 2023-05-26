from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Sequence

from pydantic import Extra, root_validator

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
    client: Client
    top_n: int = 3
    model: str = "rerank-english-v2.0"

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        cohere_api_key = get_from_dict_or_env(
            values, "cohere_api_key", "COHERE_API_KEY"
        )
        try:
            import cohere

            values["client"] = cohere.Client(cohere_api_key)
        except ImportError:
            raise ImportError(
                "Could not import cohere python package. "
                "Please install it with `pip install cohere`."
            )
        return values

    def compress_documents(
        self, documents: Sequence[Document], query: str
    ) -> Sequence[Document]:
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

    async def acompress_documents(
        self, documents: Sequence[Document], query: str
    ) -> Sequence[Document]:
        raise NotImplementedError
