from typing import List
from elasticsearch import Elasticsearch
from elasticsearch.client import MlClient


class ElasticsearchEmbeddings:
    """
    Wrapper around Elasticsearch embedding models.
    
    This class provides an interface to generate embeddings using a model deployed
    in an Elasticsearch cluster. It requires an Elasticsearch connection object
    and the model_id of the model deployed in the cluster.
    """

    def __init__(self, es_connection: Elasticsearch, model_id: str):
        """
        Initialize the ElasticsearchEmbeddings instance.

        Args:
            es_connection (Elasticsearch): An Elasticsearch connection object.
            model_id (str): The model_id of the model deployed in the Elasticsearch cluster.
        """
        self.es_connection = es_connection
        self.ml_client = MlClient(es_connection)
        self.model_id = model_id

    def _embedding_func(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for the given texts using the Elasticsearch model.

        Args:
            texts (List[str]): A list of text strings to generate embeddings for.

        Returns:
            List[List[float]]: A list of embeddings, one for each text in the input list.
        """
        response = self.ml_client.infer_trained_model(
            model_id=self.model_id,
            docs=[{"text": text} for text in texts]
        )
        embeddings = [doc["results"][0]["vector"] for doc in response["docs"]]
        return embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of documents.

        Args:
            texts (List[str]): A list of document text strings to generate embeddings for.

        Returns:
            List[List[float]]: A list of embeddings, one for each document in the input list.
        """
        return self._embedding_func(texts)

    def embed_query(self, text: str) -> List[float]:
        """
        Generate an embedding for a single query text.

        Args:
            text (str): The query text to generate an embedding for.

        Returns:
            List[float]: The embedding for the input query text.
        """
        return self._embedding_func([text])[0]
