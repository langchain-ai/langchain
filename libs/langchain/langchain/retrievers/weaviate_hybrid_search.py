from __future__ import annotations

from typing import Any, Dict, List, Optional, cast
from uuid import uuid4

from langchain_core.documents import Document
from langchain_core.pydantic_v1 import root_validator
from langchain_core.retrievers import BaseRetriever

from langchain.callbacks.manager import CallbackManagerForRetrieverRun


class WeaviateHybridSearchRetriever(BaseRetriever):
    """`Weaviate hybrid search` retriever.

    See the documentation:
      https://weaviate.io/blog/hybrid-search-explained
    """

    client: Any
    """keyword arguments to pass to the Weaviate client."""
    index_name: str
    """The name of the index to use."""
    text_key: str
    """The name of the text key to use."""
    alpha: float = 0.5
    """The weight of the text key in the hybrid search."""
    k: int = 4
    """The number of results to return."""
    attributes: List[str]
    """The attributes to return in the results."""
    create_schema_if_missing: bool = True
    """Whether to create the schema if it doesn't exist."""

    @root_validator(pre=True)
    def validate_client(
        cls,
        values: Dict[str, Any],
    ) -> Dict[str, Any]:
        try:
            import weaviate
        except ImportError:
            raise ImportError(
                "Could not import weaviate python package. "
                "Please install it with `pip install weaviate-client`."
            )
        if not isinstance(values["client"], weaviate.Client):
            client = values["client"]
            raise ValueError(
                f"client should be an instance of weaviate.Client, got {type(client)}"
            )
        if values.get("attributes") is None:
            values["attributes"] = []

        cast(List, values["attributes"]).append(values["text_key"])

        if values.get("create_schema_if_missing", True):
            class_obj = {
                "class": values["index_name"],
                "properties": [{"name": values["text_key"], "dataType": ["text"]}],
                "vectorizer": "text2vec-openai",
            }

            if not values["client"].schema.exists(values["index_name"]):
                values["client"].schema.create_class(class_obj)

        return values

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    # added text_key
    def add_documents(self, docs: List[Document], **kwargs: Any) -> List[str]:
        """Upload documents to Weaviate."""
        from weaviate.util import get_valid_uuid

        with self.client.batch as batch:
            ids = []
            for i, doc in enumerate(docs):
                metadata = doc.metadata or {}
                data_properties = {self.text_key: doc.page_content, **metadata}

                # If the UUID of one of the objects already exists
                # then the existing objectwill be replaced by the new object.
                if "uuids" in kwargs:
                    _id = kwargs["uuids"][i]
                else:
                    _id = get_valid_uuid(uuid4())

                batch.add_data_object(data_properties, self.index_name, _id)
                ids.append(_id)
        return ids

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        where_filter: Optional[Dict[str, object]] = None,
        score: bool = False,
        hybrid_search_kwargs: Optional[Dict[str, object]] = None,
    ) -> List[Document]:
        """Look up similar documents in Weaviate.

        query: The query to search for relevant documents
         of using weviate hybrid search.

        where_filter: A filter to apply to the query.
            https://weaviate.io/developers/weaviate/guides/querying/#filtering

        score: Whether to include the score, and score explanation
            in the returned Documents meta_data.

        hybrid_search_kwargs: Used to pass additional arguments
         to the .with_hybrid() method.
            The primary uses cases for this are:
            1)  Search specific properties only -
                specify which properties to be used during hybrid search portion.
                Note: this is not the same as the (self.attributes) to be returned.
                Example - hybrid_search_kwargs={"properties": ["question", "answer"]}
            https://weaviate.io/developers/weaviate/search/hybrid#selected-properties-only

            2) Weight boosted searched properties -
                Boost the weight of certain properties during the hybrid search portion.
                Example - hybrid_search_kwargs={"properties": ["question^2", "answer"]}
            https://weaviate.io/developers/weaviate/search/hybrid#weight-boost-searched-properties

            3) Search with a custom vector - Define a different vector
                to be used during the hybrid search portion.
                Example - hybrid_search_kwargs={"vector": [0.1, 0.2, 0.3, ...]}
            https://weaviate.io/developers/weaviate/search/hybrid#with-a-custom-vector

            4) Use Fusion ranking method
                Example - from weaviate.gql.get import HybridFusion
                hybrid_search_kwargs={"fusion": fusion_type=HybridFusion.RELATIVE_SCORE}
            https://weaviate.io/developers/weaviate/search/hybrid#fusion-ranking-method
        """
        query_obj = self.client.query.get(self.index_name, self.attributes)
        if where_filter:
            query_obj = query_obj.with_where(where_filter)

        if score:
            query_obj = query_obj.with_additional(["score", "explainScore"])

        if hybrid_search_kwargs is None:
            hybrid_search_kwargs = {}

        result = (
            query_obj.with_hybrid(query, alpha=self.alpha, **hybrid_search_kwargs)
            .with_limit(self.k)
            .do()
        )
        if "errors" in result:
            raise ValueError(f"Error during query: {result['errors']}")

        docs = []

        for res in result["data"]["Get"][self.index_name]:
            text = res.pop(self.text_key)
            docs.append(Document(page_content=text, metadata=res))
        return docs
