from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from langchain.utils import get_from_env

if TYPE_CHECKING:
    from elasticsearch import Elasticsearch
    from elasticsearch.client import MlClient

from langchain.schema.embeddings import Embeddings


class ElasticsearchEmbeddings(Embeddings):
    """Elasticsearch embedding models.

    This class provides an interface to generate embeddings using a model deployed
    in an Elasticsearch cluster. It requires an Elasticsearch connection object
    and the model_id of the model deployed in the cluster.

    In Elasticsearch you need to have an embedding model loaded and deployed.
    - https://www.elastic.co/guide/en/elasticsearch/reference/current/infer-trained-model.html
    - https://www.elastic.co/guide/en/machine-learning/current/ml-nlp-deploy-models.html
    """  # noqa: E501

    def __init__(
        self,
        client: MlClient,
        model_id: str,
        *,
        input_field: str = "text_field",
    ):
        """
        Initialize the ElasticsearchEmbeddings instance.

        Args:
            client (MlClient): An Elasticsearch ML client object.
            model_id (str): The model_id of the model deployed in the Elasticsearch
                cluster.
            input_field (str): The name of the key for the input text field in the
                document. Defaults to 'text_field'.
        """
        self.client = client
        self.model_id = model_id
        self.input_field = input_field

    @classmethod
    def from_credentials(
        cls,
        model_id: str,
        *,
        es_cloud_id: Optional[str] = None,
        es_user: Optional[str] = None,
        es_password: Optional[str] = None,
        input_field: str = "text_field",
    ) -> ElasticsearchEmbeddings:
        """Instantiate embeddings from Elasticsearch credentials.

        Args:
            model_id (str): The model_id of the model deployed in the Elasticsearch
                cluster.
            input_field (str): The name of the key for the input text field in the
                document. Defaults to 'text_field'.
            es_cloud_id: (str, optional): The Elasticsearch cloud ID to connect to.
            es_user: (str, optional): Elasticsearch username.
            es_password: (str, optional): Elasticsearch password.

        Example:
            .. code-block:: python

                from langchain.embeddings import ElasticsearchEmbeddings

                # Define the model ID and input field name (if different from default)
                model_id = "your_model_id"
                # Optional, only if different from 'text_field'
                input_field = "your_input_field"

                # Credentials can be passed in two ways. Either set the env vars
                # ES_CLOUD_ID, ES_USER, ES_PASSWORD and they will be automatically
                # pulled in, or pass them in directly as kwargs.
                embeddings = ElasticsearchEmbeddings.from_credentials(
                    model_id,
                    input_field=input_field,
                    # es_cloud_id="foo",
                    # es_user="bar",
                    # es_password="baz",
                )

                documents = [
                    "This is an example document.",
                    "Another example document to generate embeddings for.",
                ]
                embeddings_generator.embed_documents(documents)
        """
        try:
            from elasticsearch import Elasticsearch
            from elasticsearch.client import MlClient
        except ImportError:
            raise ImportError(
                "elasticsearch package not found, please install with 'pip install "
                "elasticsearch'"
            )

        es_cloud_id = es_cloud_id or get_from_env("es_cloud_id", "ES_CLOUD_ID")
        es_user = es_user or get_from_env("es_user", "ES_USER")
        es_password = es_password or get_from_env("es_password", "ES_PASSWORD")

        # Connect to Elasticsearch
        es_connection = Elasticsearch(
            cloud_id=es_cloud_id, basic_auth=(es_user, es_password)
        )
        client = MlClient(es_connection)
        return cls(client, model_id, input_field=input_field)

    @classmethod
    def from_es_connection(
        cls,
        model_id: str,
        es_connection: Elasticsearch,
        input_field: str = "text_field",
    ) -> ElasticsearchEmbeddings:
        """
        Instantiate embeddings from an existing Elasticsearch connection.

        This method provides a way to create an instance of the ElasticsearchEmbeddings
        class using an existing Elasticsearch connection. The connection object is used
        to create an MlClient, which is then used to initialize the
        ElasticsearchEmbeddings instance.

        Args:
        model_id (str): The model_id of the model deployed in the Elasticsearch cluster.
        es_connection (elasticsearch.Elasticsearch): An existing Elasticsearch
        connection object. input_field (str, optional): The name of the key for the
        input text field in the document. Defaults to 'text_field'.

        Returns:
        ElasticsearchEmbeddings: An instance of the ElasticsearchEmbeddings class.

        Example:
            .. code-block:: python

                from elasticsearch import Elasticsearch

                from langchain.embeddings import ElasticsearchEmbeddings

                # Define the model ID and input field name (if different from default)
                model_id = "your_model_id"
                # Optional, only if different from 'text_field'
                input_field = "your_input_field"

                # Create Elasticsearch connection
                es_connection = Elasticsearch(
                    hosts=["localhost:9200"], http_auth=("user", "password")
                )

                # Instantiate ElasticsearchEmbeddings using the existing connection
                embeddings = ElasticsearchEmbeddings.from_es_connection(
                    model_id,
                    es_connection,
                    input_field=input_field,
                )

                documents = [
                    "This is an example document.",
                    "Another example document to generate embeddings for.",
                ]
                embeddings_generator.embed_documents(documents)
        """
        # Importing MlClient from elasticsearch.client within the method to
        # avoid unnecessary import if the method is not used
        from elasticsearch.client import MlClient

        # Create an MlClient from the given Elasticsearch connection
        client = MlClient(es_connection)

        # Return a new instance of the ElasticsearchEmbeddings class with
        # the MlClient, model_id, and input_field
        return cls(client, model_id, input_field=input_field)

    def _embedding_func(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for the given texts using the Elasticsearch model.

        Args:
            texts (List[str]): A list of text strings to generate embeddings for.

        Returns:
            List[List[float]]: A list of embeddings, one for each text in the input
                list.
        """
        response = self.client.infer_trained_model(
            model_id=self.model_id, docs=[{self.input_field: text} for text in texts]
        )

        embeddings = [doc["predicted_value"] for doc in response["inference_results"]]
        return embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of documents.

        Args:
            texts (List[str]): A list of document text strings to generate embeddings
                for.

        Returns:
            List[List[float]]: A list of embeddings, one for each document in the input
                list.
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
