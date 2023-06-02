from __future__ import annotations

import os
import traceback
import uuid
from typing import Any, Dict, Iterable, List, Optional

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.base import VectorStore


class Clarifai(VectorStore):
    """Wrapper around Clarifai AI platform's vector store.

    To use, you should have the ``clarifai`` python package installed.

    Example:
        .. code-block:: python

                from langchain.vectorstores import Clarifai
                from langchain.embeddings.openai import OpenAIEmbeddings

                embeddings = OpenAIEmbeddings()
                vectorstore = Clarifai("langchain_store", embeddings.embed_query)
    """

    def __init__(
        self,
        user_id: str,
        app_id: str,
        pat: Optional[str] = None,
        number_of_docs: Optional[int] = None,
        api_base: Optional[str] = None,
    ) -> None:
        """Initialize with Clarifai client."""
        try:
            pass
        except ImportError:
            raise ValueError(
                "Could not import clarifai python package. " "Please install it with `pip install clarifai`."
            )

        if pat is None:
            if "CLARIFAI_PAT_KEY" not in os.environ:
                raise ValueError(
                    "Could not find CLARIFAI_PAT in your environment. "
                    "Please set that env variable with a valid personal access token from https://clarifai.com/settings/security."
                )
            pat = os.environ["CLARIFAI_PAT"]

        from clarifai.auth.helper import DEFAULT_BASE, ClarifaiAuthHelper
        from clarifai.client import create_stub

        if api_base is None:
            api_base = DEFAULT_BASE

        auth = ClarifaiAuthHelper(user_id=user_id, app_id=app_id, pat=pat, base=api_base)
        stub = create_stub(auth)
        userDataObject = auth.get_user_app_id_proto()

        self._stub = stub
        self._auth = auth
        self._userDataObject = userDataObject
        self._number_of_docs = number_of_docs

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts (Iterable[str]): Texts to add to the vectorstore.
            metadatas (Optional[List[dict]], optional): Optional list of metadatas.
            ids (Optional[List[str]], optional): Optional list of IDs.

        Returns:
            List[str]: List of IDs of the added texts.
        """
        # TODO: Handle the case where the user doesn't provide ids on the Collection
        if ids is None:
            ids = [str(uuid.uuid1()) for _ in texts]
        embeddings = None
        if self._embedding_function is not None:
            embeddings = self._embedding_function.embed_documents(list(texts))
        self._collection.add(metadatas=metadatas, embeddings=embeddings, documents=texts, ids=ids)
        return ids

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
        namespace: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Run similarity search with Clarifai.

        Args:
            query (str): Query text to search for.
            k (int): Number of results to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List[Document]: List of documents most simmilar to the query text.
        """
        import requests
        from clarifai_grpc.grpc.api import resources_pb2, service_pb2
        from clarifai_grpc.grpc.api.status import status_code_pb2
        from google.protobuf import json_format

        if self._number_of_docs is not None:
            k = self._number_of_docs

        # traceback.print_stack()
        post_annotations_searches_response = self._stub.PostAnnotationsSearches(
            service_pb2.PostAnnotationsSearchesRequest(
                user_app_id=self._userDataObject,
                searches=[
                    resources_pb2.Search(
                        query=resources_pb2.Query(
                            ranks=[
                                resources_pb2.Rank(
                                    annotation=resources_pb2.Annotation(
                                        data=resources_pb2.Data(
                                            text=resources_pb2.Text(raw=query),
                                        )
                                    )
                                )
                            ]
                        )
                    )
                ],
                pagination=service_pb2.Pagination(page=1, per_page=self._number_of_docs),
            )
        )

        if post_annotations_searches_response.status.code != status_code_pb2.SUCCESS:
            raise Exception("Post searches failed, status: " + post_annotations_searches_response.status.description)

        hits = post_annotations_searches_response.hits

        docs_and_scores = []
        for hit in hits:
            metadata = json_format.MessageToDict(hit.input.data.metadata)
            request = requests.get(hit.input.data.text.url)

            # override encoding by real educated guess as provided by chardet
            request.encoding = request.apparent_encoding
            requested_text = request.text

            print(
                "\tScore %.2f for annotation: %s off input: %s, text: %s"
                % (
                    hit.score,
                    hit.annotation.id,
                    hit.input.id,
                    requested_text[:125],
                )
            )

            docs_and_scores.append((Document(page_content=requested_text, metadata=metadata), hit.score))

        return docs_and_scores

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
        namespace: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return pinecone documents most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Dictionary of argument(s) to filter on metadata
            namespace: Namespace to search in. Default will search in '' namespace.

        Returns:
            List of Documents most similar to the query and score for each
        """
        docs_and_scores = self.similarity_search_with_score(query, **kwargs)
        return [doc for doc, _ in docs_and_scores]

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Optional[Embeddings] = None,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        persist_directory: Optional[str] = None,
        **kwargs: Any,
    ) -> Clarifai:
        """Create a Clarifai vectorstore from a raw documents.

        If a persist_directory is specified, the collection will be persisted there.
        Otherwise, the data will be ephemeral in-memory.

        Args:
            collection_name (str): Name of the collection to create.
            persist_directory (Optional[str]): Directory to persist the collection.
            documents (List[Document]): List of documents to add.
            embedding (Optional[Embeddings]): Embedding function. Defaults to None.
            metadatas (Optional[List[dict]]): List of metadatas. Defaults to None.
            ids (Optional[List[str]]): List of document IDs. Defaults to None.

        Returns:
            Clarifai: Clarifai vectorstore.
        """
        raise NotImplementedError("not yet ready")
        chroma_collection = cls(
            collection_name=collection_name,
            embedding_function=embedding,
            persist_directory=persist_directory,
        )
        chroma_collection.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        return chroma_collection

    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        embedding: Optional[Embeddings] = None,
        ids: Optional[List[str]] = None,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        persist_directory: Optional[str] = None,
        **kwargs: Any,
    ) -> Clarifai:
        """Create a Clarifai vectorstore from a list of documents.

        If a persist_directory is specified, the collection will be persisted there.
        Otherwise, the data will be ephemeral in-memory.

        Args:
            collection_name (str): Name of the collection to create.
            persist_directory (Optional[str]): Directory to persist the collection.
            documents (List[Document]): List of documents to add to the vectorstore.
            embedding (Optional[Embeddings]): Embedding function. Defaults to None.

        Returns:
            Clarifai: Clarifai vectorstore.
        """
        raise NotImplementedError("not yet ready")
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return cls.from_texts(
            texts=texts,
            embedding=embedding,
            metadatas=metadatas,
            ids=ids,
            collection_name=collection_name,
            persist_directory=persist_directory,
        )
