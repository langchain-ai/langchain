
from typing import Any, List, Optional, Iterable, Tuple
from langchain.schema import BaseRetriever, Document
import json
import logging

def _error_msg(response):
    """Returns an error message constructed from the passed http response."""
    return f"(code {response.status_code}, reason {response.reason}, details {response.text})"

class VectaraRetriever(BaseRetriever):
    def __init__(
        self,
        alpha: float = 0.025,   # called "lambda" in Vectara, but changed here to alpha since its a reserved word in python
        k: int = 5,
        filter: Optional[dict] = None,
    ):
        self.k = k
        self.alpha = alpha
        self.filter = filter
    """
    Implements a Vectara retriever with integrated Hybrid search functionality
    """

    def get_relevant_documents(
        self, query: str
    ) -> List[Document]:
        """Look up similar documents in Vectara."""

        response = self._session.post(
            headers=self._get_post_headers(),
            url="https://api.vectara.io/v1/query",
            data=json.dumps(
                {
                    "query": [
                        {
                            "query": query,
                            "num_results": self.k,
                            "corpus_key": [
                                {
                                    "customer_id": self._customer_id,
                                    "corpus_id": self._corpus_id,
                                    "metadataFilter": filter,
                                    "lexical_interpolation_config": {
                                        "lambda": self.alpha
                                    },
                                }
                            ],
                            "reranking_config": {"rerankerId": 272725717},
                        }
                    ]
                }
            ),
            timeout=10,
        )

        if response.status_code != 200:
            logging.error("Query failed %s", _error_msg(response))
            return []

        result = response.json()
        docs = [Document(page_content=x["text"], metadata=x['metadata']) for x in result["responseSet"][0]["response"]]

        return docs
    