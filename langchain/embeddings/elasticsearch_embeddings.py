from typing import List
from elasticsearch import Elasticsearch
from elasticsearch.client import MlClient


class ElasticsearchEmbeddings:
    """
    ElasticsearchEmbeddings is a class that wraps around Elasticsearch's
    text embedding models to generate embeddings for the given text.

    Attributes:
        model_id (str): The model_id of the running Elasticsearch model used for generating embeddings.
        es_client (Elasticsearch): An Elasticsearch client instance for connecting to the Elasticsearch cluster.
        ml_client (MlClient): An Elasticsearch MlClient instance for interacting with machine learning models.
    """

    def __init__(self, model_id: str, es_connection: Elasticsearch):
        """
        Initializes an instance of ElasticsearchEmbeddings.

        Args:
            model_id (str): The model_id of the running Elasticsearch model used for generating embeddings.
            es_connection (Elasticsearch): An Elasticsearch client instance for connecting to the Elasticsearch cluster.
        """
        self.model_id = model_id
        self.es_client = es_connection
        self.ml_client = MlClient(self.es_client)

    def embed_text(self, text: str) -> List[float]:
        """
        Generates an embedding for the input text using the specified Elasticsearch model.

        Args:
            text (str): The input text to generate an embedding for.

        Returns:
            List[float]: A list of floating-point numbers representing the embedding of the input text.
        """
        response = self.ml_client.infer_trained_model(
            model_id=self.model_id,
            docs=[{"text_field": text}],
        )

        return response["results"][0]["embedding"]

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embeddings for a list of input texts using the specified Elasticsearch model.

        Args:
            texts (List[str]): A list of input texts to generate embeddings for.

        Returns:
            List[List[float]]: A list of lists, where each inner list is a list of floating-point numbers representing
                               the embedding of the corresponding input text.
        """
        response = self.ml_client.infer_trained_model(
            model_id=self.model_id,
            docs=[{"text_field": text} for text in texts],
        )

        return [result["embedding"] for result in response["results"]]
