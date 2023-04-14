"""Vertex Matching Engine implementation of the vector store."""

from google.cloud import aiplatform
from google.cloud import storage
from google.oauth2 import service_account
from typing import Any, Iterable, List, Optional, Union
import uuid

from langchain.docstore.document import Document
from langchain.embeddings import TensorflowHubEmbeddings
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.base import VectorStore


class MatchingEngine(VectorStore):
    """Vertex Matching Engine implementation of the vector store."""

    def __init__(self,
        project_id: str,
        region: str,
        gcs_bucket_uri: str,
        index_name: str, # TODO document and tell the user that they need to run some lines to create the matching engine -> notebook de quickstart
        endpoint_name: str, # TODO document and tell the user that they need to run some lines to create the matching engine -> notebook de quickstart
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
        self.project_id = project_id
        self.region = region
        self.gcs_client = None
        self.gcs_bucket_uri = gcs_bucket_uri
        self.embedder = embedder

        self.credentials = self._create_credentials_from_file(json_credentials_path)
        self._init_aiplatform(project_id, region, gcs_bucket_uri)
        self.index = self._create_index_by_name(index_name)
        self.endpoint = self._create_endpoint_by_name(endpoint_name)
        

    def _create_credentials_from_file(self, json_credentials_path: Union[str, None]) -> service_account.Credentials:
        """TODO docs"""

        credentials = None
        if json_credentials_path is not None:
            credentials = service_account.Credentials.from_service_account_file(json_credentials_path)

        return credentials
    
    def _init_aiplatform(self, project_id: str, region: str, gcs_bucket_uri: str) -> None:
        """TODO add docs"""
        aiplatform.init(
            project=project_id, 
            location=region, 
            staging_bucket=gcs_bucket_uri,
            credentials=self.credentials
        )

    def _create_index_by_name(self, index_name: str) -> "aiplatform.MatchingEngineIndex":
        """TODO add docs"""
        return aiplatform.MatchingEngineIndex(
            index_name,
            project=self.project_id,
            location=self.region,
            credentials=self.credentials
        )
    
    def _create_endpoint_by_name(self, endpoint_name: str) -> "aiplatform.MatchingEngineIndexEndpoint":
        """TODO add docs"""
        return aiplatform.MatchingEngineIndexEndpoint(
            endpoint_name,
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

        result_str = "\n".join(jsons)

        filename = f"{uuid.uuid4()}.json"
        self._upload_to_gcs(result_str, filename)

        self.index = self.index.update_embeddings(
            contents_delta_uri=f"{self.gcs_bucket_uri}/indexes/{filename}",
        )

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

        embedding_query = self.embedder.embed_documents([query])

        # I'm only getting the first one because queries receives an array and the similarity_search 
        # method only recevies one query. This means that the match method will always return an array
        # with only one element.

        index_id = None

        for index in self.endpoint.deployed_indexes:
            if index.index == self.index.resource_name:
                index_id = index.id
                break
        
        if index_id == None:
            raise ValueError(f"No index with id {self.index.resource_name} deployed on enpoint {self.endpoint.display_name}.")

        response = self.endpoint.match(
            deployed_index_id=index_id,
            queries=embedding_query,
            num_neighbors=k,
        )[0]

        results = []

        for doc in response:
            page_content = self._download_from_gcs(f"documents/{doc.id}")
            results.append(Document(page_content=page_content))
            
        return results
    
    def _download_from_gcs(self, gcs_location: str) -> str:
        """TODO add docs"""
        client = self._get_gcs_client()
        bucket = client.get_bucket(self.gcs_bucket_uri)
        blob = bucket.blob(gcs_location)
        return blob.download_as_string()

# TODO delete this after testing
if __name__ == "__main__":
    me = MatchingEngine(
        project_id="scafati-joonix",
        region="us-central1",
        gcs_bucket_uri="gs://langchain-integration",
        index_name="glove_100_1_langchain",
        endpoint_name="tree_ah_glove_deployed_langchain"
    )