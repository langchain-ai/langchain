from __future__ import annotations

from typing import Any, List

from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.schema import BaseRetriever, Document


class SecFilingsRetriever(BaseRetriever):
    """
        Retriever for SEC filings, powered by Kay.ai and Cybersyn.

        To work properly, expects you to have KAY_API_KEY env variable set.
        You can get one for free at https://kay.ai/.
    """

    client: Any
    num_contexts: int

    @classmethod
    def create(cls, num_contexts: int = 3) -> SecFilingsRetriever:
        """
        Create a KayRetriever given a Kay dataset id and a list of datasources.

        Args:
            num_contexts: The number of documents to retrieve on each query.
                Defaults to 3.
        """
        try:
            from kay.rag.retrievers import KayRetriever
        except ImportError:
            raise ImportError(
                "Could not import kay python package. Please install it with "
                "`pip install kay`.",
            )

        client = KayRetriever(dataset_id="company", data_types=["10-K", "10-Q"])
        return cls(client=client, num_contexts=num_contexts)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        ctxs = self.client.query(query=query, num_context=self.num_contexts)
        docs = []
        for ctx in ctxs:
            page_content = ctx.pop("chunk_embed_text", None)
            if page_content is None:
                continue
            docs.append(Document(page_content=page_content, metadata={**ctx}))
        return docs
 
