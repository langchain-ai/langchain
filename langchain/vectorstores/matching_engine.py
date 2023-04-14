"""Vertex Matching Engine implementation of the vector store."""

from google.cloud import aiplatform
from google.cloud import storage
from google.oauth2 import service_account
from typing import Any, Iterable, List, Optional, Type, TypeVar, Union
import uuid
import logging

from langchain.docstore.document import Document
from langchain.embeddings import TensorflowHubEmbeddings
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.base import VectorStore

logger = logging.getLogger()


class MatchingEngine(VectorStore):
    """Vertex Matching Engine implementation of the vector store."""

    def __init__(self,
        project_id: str,
        region: str,
        gcs_bucket_uri: str,
        index_id: str, # TODO document and tell the user that they need to run some lines to create the matching engine -> notebook de quickstart
        endpoint_id: str, # TODO document and tell the user that they need to run some lines to create the matching engine -> notebook de quickstart
        json_credentials_path: Union[str, None] = None,
        embedder: Embeddings = TensorflowHubEmbeddings(model_url="https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")
    ):
        """Vertex Matching Engine implementation of the vector store. 
            TODO: create docs for this module: https://github.com/hwchase17/langchain/blob/master/docs/modules/indexes/vectorstores.rst
            TODO: preconditio for this class the index and the endpoint must exist
            TODO: add aiplatform and storage dependencies to poetry
            Attributes:
                project_id: The GCS project id.
                region: The default location making the API calls. It must have the 
                same location 
                gcs_bucket_uri: .
        """
        super().__init__()
        logger.debug(f"Constructor.")
        self.project_id = project_id
        self.region = region
        self.gcs_client = None
        self.gcs_bucket_uri = gcs_bucket_uri
        self.embedder = embedder

        self.credentials = self._create_credentials_from_file(json_credentials_path)
        self._init_aiplatform(project_id, region, gcs_bucket_uri)
        self.index = self._create_index_by_id(index_id)
        self.endpoint = self._create_endpoint_by_id(endpoint_id)
        

    def _create_credentials_from_file(self, json_credentials_path: Union[str, None]) -> service_account.Credentials:
        """TODO docs"""

        credentials = None
        if json_credentials_path is not None:
            credentials = service_account.Credentials.from_service_account_file(json_credentials_path)

        return credentials
    
    def _init_aiplatform(self, project_id: str, region: str, gcs_bucket_uri: str) -> None:
        """TODO add docs"""
        logger.debug(f"Initializing AI Platform for project {project_id} on "
                     f"{region} and for {gcs_bucket_uri}.")
        aiplatform.init(
            project=project_id, 
            location=region, 
            staging_bucket=gcs_bucket_uri,
            credentials=self.credentials
        )

    def _create_index_by_id(self, index_id: str) -> "aiplatform.MatchingEngineIndex":
        """TODO add docs"""
        logger.debug(f"Creating matching engine index with id {index_id}.")
        return aiplatform.MatchingEngineIndex(
            index_name=index_id,
            project=self.project_id,
            location=self.region,
            credentials=self.credentials
        )
    
    def _create_endpoint_by_id(self, endpoint_id: str) -> "aiplatform.MatchingEngineIndexEndpoint":
        """TODO add docs"""
        logger.debug(f"Creating endpoint with id {endpoint_id}.")
        return aiplatform.MatchingEngineIndexEndpoint(
            index_endpoint_name=endpoint_id,
            project=self.project_id,
            location=self.region,
            credentials=self.credentials,
        )

    
    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            kwargs: vectorstore specific parameters

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        logger.debug(f"Embedding documents.")
        embeddings = self.embedder.embed_documents(texts)
        jsons = []
        ids = []
        # TODO improve with async
        for embedding, text in zip(embeddings, texts):
            id = uuid.uuid4()
            ids.append(id)
            jsons.append({
                "id": id,
                "embedding": embedding
            })
            self._upload_to_gcs(text, f"documents/{id}")

        logger.debug(f"Uploaded {len(ids)} documents to GCS.")

        result_str = "\n".join(jsons)

        filename = f"{uuid.uuid4()}.json"
        self._upload_to_gcs(result_str, filename)
        logger.debug(f"Uploaded updated json with embeddings to {self.gcs_bucket_uri}/indexes/{filename}.")

        self.index = self.index.update_embeddings(
            contents_delta_uri=f"{self.gcs_bucket_uri}/indexes/{filename}",
        )

        logger.debug(f"Updated index with new configuration.")

        return ids
    
    def _upload_to_gcs(self, data: str, gcs_location: str) -> None:
        """TODO add docs"""
        client = self._get_gcs_client()
        bucket = client.get_bucket(self.gcs_bucket_uri)
        blob = bucket.blob(gcs_location)
        blob.upload_from_string(data)

    def _get_gcs_client(self) -> storage.Client:
        """TODO add docs"""
        if self.gcs_client is None:            
            self.gcs_client = storage.Client(
                credentials=self.credentials, 
                project=self.project_id
            )

        return self.gcs_client

    def similarity_search(
        self, 
        query: str, 
        k: int = 4, 
        **kwargs: Any
    ) -> List[Document]:
        """Return docs most similar to query."""

        logger.debug(f"Embedding query {query}.")
        embedding_query = self.embedder.embed_documents([query])

        index_id = None

        for index in self.endpoint.deployed_indexes:
            if index.index == self.index.resource_name:
                index_id = index.id
                break
        
        if index_id == None:
            raise ValueError(f"No index with id {self.index.resource_name} deployed on enpoint {self.endpoint.display_name}.")

        # I'm only getting the first one because queries receives an array and the similarity_search 
        # method only recevies one query. This means that the match method will always return an array
        # with only one element.
        response = self.endpoint.match(
            deployed_index_id=index_id,
            queries=embedding_query,
            num_neighbors=k,
        )

        if len(response) == 0:
            return []

        logger.debug(f"Found {len(response)} matches for the query {query}.")

        results = []

        for doc in response:
            page_content = self._download_from_gcs(f"documents/{doc.id}")
            results.append(Document(page_content=page_content))

        logger.debug(f"Downloaded documents for query.")
            
        return results
    
    def _download_from_gcs(self, gcs_location: str) -> str:
        """TODO add docs"""
        client = self._get_gcs_client()
        bucket = client.get_bucket(self.gcs_bucket_uri)
        blob = bucket.blob(gcs_location)
        return blob.download_as_string()
    
    @classmethod
    def from_texts(
        cls: Type["MatchingEngine"], 
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None, 
        project_id: str = None,
        region: str = None,
        gcs_bucket_uri: str = None,
        index_name: str = None,
        endpoint_name: str = None,
        **kwargs: Any,
    ) -> "MatchingEngine":
        """Return VectorStore initialized from texts and embeddings."""
        matching_engine = cls(
            project_id=project_id,
            region=region,
            gcs_bucket_uri=gcs_bucket_uri,
            index_name=index_name,
            endpoint_id=endpoint_name,
            embedder=embedding
        )

        matching_engine.add_texts(texts=texts, metadatas=metadatas)
        return matching_engine
    

# TODO delete this after testing
if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
    me = MatchingEngine(
        project_id="scafati-joonix",
        region="us-central1",
        gcs_bucket_uri="gs://langchain-integration",
        index_id="1419223220854194176",
        endpoint_id="4789041642034167808"
    )

    print(me.similarity_search("Cristian Castro"))