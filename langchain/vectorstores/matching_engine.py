"""Vertex Matching Engine implementation of the vector store."""

import json
from typing import Any, Dict, Iterable, List, Optional, Type, TypeVar, Union
from langchain.vectorstores.base import VectorStore
from langchain.docstore.document import Document
import asyncio
from functools import partial
import uuid
from langchain.embeddings import Embeddings, TensorflowHubEmbeddings
from google.cloud import storage
from oauth2client.service_account import ServiceAccountCredentials
import os

from google.cloud import aiplatform

class MatchingEngine(VectorStore):

    def __init__(self,
        project_id: str,
        region: str,
        gcs_bucket_uri: str,
        index_id: str, # TODO document and tell the user that they need to run some lines to create the matching engine -> notebook de quickstart
        endpoint_id: str, # TODO document and tell the user that they need to run some lines to create the matching engine -> notebook de quickstart
        json_credentials_path: Union[str, None] = None,
        embedder: Embeddings = TensorflowHubEmbeddings(model_url="https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")
    ):
        """
            TODO: fininsh docs
            TODO: preconditio for this class the index and the endpoint must exist
            Attributes:
                project_id: The GCS project id.
                region: The default location making the API calls. It must have the 
                same location 
                gcs_bucket_uri: .
        """
        self._init_aiplatform(project_id, region, gcs_bucket_uri)
        self.index = self._find_index_by_id(index_id)
        self.embedder = embedder
        self.endpoint_id = endpoint_id

        self.gcs_client = None
        self.project_id = project_id
        self.json_credentials_path = json_credentials_path
        self.gcs_bucket_uri = gcs_bucket_uri
        

    def _init_aiplatform(self, project_id: str, region: str, gcs_bucket_uri: str) -> None:
        """TODO add docs"""
        aiplatform.init(
            project=project_id, 
            location=region, 
            staging_bucket=gcs_bucket_uri
        )

    def _find_index_by_id(self, index_id: str) -> "aiplatform.MatchingEngineIndex":
        """TODO add docs and change return type"""
        indexes = aiplatform.MatchingEngineIndex.deployed_indexes() 
        found_index = None
        for index in indexes: 
            if index.deployed_index_id == index_id:
                found_index = index
                break

        if found_index is None:
            raise ValueError(f"Matching Index with id {index_id} not found.")
        
        return found_index

    
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
        for embedding in embeddings:
            id = uuid.uuid4()
            ids.append(id)
            jsons.append({
                "id": id,
                "embedding": embedding
            })

        result_str = "\n".join(jsons)

        client = self._get_gcs_client()
        bucket = client.get_bucket(self.gcs_bucket_uri)
        filename = f"{uuid.uuid4()}.json"
        blob = bucket.blob(filename)
        blob.upload_from_string(result_str)

        self.index = self.index.update_embeddings(
            contents_delta_uri=f"{self.gcs_bucket_uri}/{filename}",
        )

        return ids

    def _get_gcs_client(self) -> storage.Client:
        """TODO add docs"""
        if self.gcs_client is None:
            credentials = None

            if self.json_credentials_path is not None:
                with open(self.json_credentials_path, "r") as f:
                    creds_file = json.load(f)
                    credentials = ServiceAccountCredentials.from_json_keyfile_dict(creds_file)
            
            self.gcs_client = storage.Client(
                credentials=credentials, 
                project=self.project_id
            )

        return self.gcs_client

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Return docs most similar to query."""

        [embedding_query] = self.embedder.embed_documents([query])

        response = self.index.match(
            deployed_index_id=self.endpoint_id,
            queries=embedding_query,
            num_neighbors=k,
        )

        response

        # TODO parse response

        return None