from typing import List

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from langchain_community.utilities import YouSearchAPIWrapper


class YouRetriever(BaseRetriever, YouSearchAPIWrapper):
    """`You` retriever that uses You.com's search API.
    It wraps results() to get_relevant_documents
    It uses all YouSearchAPIWrapper arguments without any change.
    """

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        import requests

        headers = {"X-API-Key": self.ydc_api_key}
        if self.endpoint_type == "search":
            return self.results(
                query=query,
                num_web_results=self.num_web_results,
                safesearch=self.safesearch,
                country=self.country,
            )
        elif self.endpoint_type == "snippet":
            results = requests.get(
                f"https://api.ydc-index.io/snippet_search?query={query}",
                headers=headers,
            ).json()
            print(results)
            return [Document(page_content=snippet) for snippet in results]
        else:
            raise RuntimeError(f"Invalid endpoint type provided {self.endpoint_type}")
