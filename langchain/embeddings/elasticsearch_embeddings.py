from typing import List

from elasticsearch import Elasticsearch
from elasticsearch.client import MlClient


class ElasticsearchEmbeddings:
    """
    Wrapper around Elasticsearch embedding models.

    This class provides an interface to generate embeddings using a model deployed
    in an Elasticsearch cluster. It requires an Elasticsearch connection object
    and the model_id of the model deployed in the cluster.

    In Elasticsearch you need to have an embedding model loaded and deployed.
    - https://www.elastic.co/guide/en/elasticsearch/reference/current/infer-trained-model.html
    - https://www.elastic.co/guide/en/machine-learning/current/ml-nlp-deploy-models.html
    """

    def __init__(
        self,
        es_connection: Elasticsearch,
        model_id: str,
        input_field: str = "text_field",
    ):
        """
        Initialize the ElasticsearchEmbeddings instance.

        Args:
            es_connection (Elasticsearch): An Elasticsearch connection object.
            model_id (str): The model_id of the model deployed in the Elasticsearch cluster.
            input_field (str): The name of the key for the input text field in the document.
                Defaults to 'text_field'.


        Example Usage:

            import os
            from elasticsearch import Elasticsearch
            from langchain.embeddings.elasticsearch_embeddings import ElasticsearchEmbeddings

            es_cloudid = os.environ.get("ES_CLOUDID")
            es_user = os.environ.get("ES_USER")
            es_pass = os.environ.get("ES_PASS")

            # Connect to Elasticsearch
            es_connection = Elasticsearch(cloud_id=es_cloudid, basic_auth=(es_user, es_pass))

            # Define the model ID and input field name (if different from default)
            model_id = "your_model_id"
            input_field = "your_input_field"  # Optional, only if different from 'text_field'

            # Initialize the ElasticsearchEmbeddings instance
            embeddings_generator = ElasticsearchEmbeddings(es_connection, model_id, input_field)

            # Generate embeddings for a list of documents
            documents = [
                "This is an example document.",
                "Another example document to generate embeddings for.",
            ]
            document_embeddings = embeddings_generator.embed_documents(documents)

            # Print the generated document embeddings
            for i, doc_embedding in enumerate(document_embeddings):
                print(f"Embedding for document {i + 1}: {doc_embedding}")

            # Generate an embedding for a single query text
            query_text = "What is the meaning of life?"
            query_embedding = embeddings_generator.embed_query(query_text)

            # Print the generated query embedding
            print(f"Embedding for query: {query_embedding}")

        """
        self.es_connection = es_connection
        self.ml_client = MlClient(es_connection)
        self.model_id = model_id
        self.input_field = input_field

    def _embedding_func(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for the given texts using the Elasticsearch model.

        Args:
            texts (List[str]): A list of text strings to generate embeddings for.

        Returns:
            List[List[float]]: A list of embeddings, one for each text in the input list.
        """
        response = self.ml_client.infer_trained_model(
            model_id=self.model_id, docs=[{self.input_field: text} for text in texts]
        )

        embeddings = [doc["predicted_value"] for doc in response["inference_results"]]
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
