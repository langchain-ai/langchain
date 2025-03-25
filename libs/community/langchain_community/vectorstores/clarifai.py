from __future__ import annotations

import logging
import os
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterable, List, Optional, Tuple

import requests
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

logger = logging.getLogger(__name__)


class Clarifai(VectorStore):
    """`Clarifai AI` vector store.

    To use, you should have the ``clarifai`` python SDK package installed.

    Example:
        .. code-block:: python

                from langchain_community.vectorstores import Clarifai

                clarifai_vector_db = Clarifai(
                        user_id=USER_ID,
                        app_id=APP_ID,
                        number_of_docs=NUMBER_OF_DOCS,
                        )
    """

    def __init__(
        self,
        user_id: Optional[str] = None,
        app_id: Optional[str] = None,
        number_of_docs: Optional[int] = 4,
        pat: Optional[str] = None,
        token: Optional[str] = None,
        api_base: Optional[str] = "https://api.clarifai.com",
    ) -> None:
        """Initialize with Clarifai client.

        Args:
            user_id (Optional[str], optional): User ID. Defaults to None.
            app_id (Optional[str], optional): App ID. Defaults to None.
            pat (Optional[str], optional): Personal access token. Defaults to None.
            token (Optional[str], optional): Session token. Defaults to None.
            number_of_docs (Optional[int], optional): Number of documents to return
            during vector search. Defaults to None.
            api_base (Optional[str], optional): API base. Defaults to None.

        Raises:
            ValueError: If user ID, app ID or personal access token is not provided.
        """
        _user_id = user_id or os.environ.get("CLARIFAI_USER_ID")
        _app_id = app_id or os.environ.get("CLARIFAI_APP_ID")
        if _user_id is None or _app_id is None:
            raise ValueError(
                "Could not find CLARIFAI_USER_ID "
                "or CLARIFAI_APP_ID in your environment. "
                "Please set those env variables with a valid user ID, app ID"
            )
        self._number_of_docs = number_of_docs

        try:
            from clarifai.client.search import Search
        except ImportError as e:
            raise ImportError(
                "Could not import clarifai python package. "
                "Please install it with `pip install clarifai`."
            ) from e

        self._auth = Search(
            user_id=_user_id,
            app_id=_app_id,
            top_k=number_of_docs,
            pat=pat,
            token=token,
            base_url=api_base,
        ).auth_helper

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add texts to the Clarifai vectorstore. This will push the text
        to a Clarifai application.
        Application use a base workflow that create and store embedding for each text.
        Make sure you are using a base workflow that is compatible with text
        (such as Language Understanding).

        Args:
            texts (Iterable[str]): Texts to add to the vectorstore.
            metadatas (Optional[List[dict]], optional): Optional list of metadatas.
            ids (Optional[List[str]], optional): Optional list of IDs.

        """
        try:
            from clarifai.client.input import Inputs
            from google.protobuf.struct_pb2 import Struct
        except ImportError as e:
            raise ImportError(
                "Could not import clarifai python package. "
                "Please install it with `pip install clarifai`."
            ) from e

        ltexts = list(texts)
        length = len(ltexts)
        assert length > 0, "No texts provided to add to the vectorstore."

        if metadatas is not None:
            assert length == len(metadatas), (
                "Number of texts and metadatas should be the same."
            )

        if ids is not None:
            assert len(ltexts) == len(ids), (
                "Number of text inputs and input ids should be the same."
            )

        input_obj = Inputs.from_auth_helper(auth=self._auth)
        batch_size = 32
        input_job_ids = []
        for idx in range(0, length, batch_size):
            try:
                batch_texts = ltexts[idx : idx + batch_size]
                batch_metadatas = (
                    metadatas[idx : idx + batch_size] if metadatas else None
                )
                if ids is None:
                    batch_ids = [uuid.uuid4().hex for _ in range(len(batch_texts))]
                else:
                    batch_ids = ids[idx : idx + batch_size]
                if batch_metadatas is not None:
                    meta_list = []
                    for meta in batch_metadatas:
                        meta_struct = Struct()
                        meta_struct.update(meta)
                        meta_list.append(meta_struct)
                input_batch = [
                    input_obj.get_text_input(
                        input_id=batch_ids[i],
                        raw_text=text,
                        metadata=meta_list[i] if batch_metadatas else None,
                    )
                    for i, text in enumerate(batch_texts)
                ]
                result_id = input_obj.upload_inputs(inputs=input_batch)
                input_job_ids.extend(result_id)
                logger.debug("Input posted successfully.")

            except Exception as error:
                logger.warning(f"Post inputs failed: {error}")
                traceback.print_exc()

        return input_job_ids

    def similarity_search_with_score(
        self,
        query: str,
        k: Optional[int] = None,
        filters: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Run similarity search with score using Clarifai.

        Args:
            query (str): Query text to search for.
            k (Optional[int]): Number of results to return. If not set,
            it'll take _number_of_docs. Defaults to None.
            filter (Optional[Dict[str, str]]): Filter by metadata.
            Defaults to None.

        Returns:
            List[Document]: List of documents most similar to the query text.
        """
        try:
            from clarifai.client.search import Search
            from clarifai_grpc.grpc.api import resources_pb2
            from google.protobuf import json_format  # type: ignore
        except ImportError as e:
            raise ImportError(
                "Could not import clarifai python package. "
                "Please install it with `pip install clarifai`."
            ) from e

        # Get number of docs to return
        top_k = k or self._number_of_docs

        search_obj = Search.from_auth_helper(auth=self._auth, top_k=top_k)
        rank = [{"text_raw": query}]
        # Add filter by metadata if provided.
        if filters is not None:
            search_metadata = {"metadata": filters}
            search_response = search_obj.query(ranks=rank, filters=[search_metadata])
        else:
            search_response = search_obj.query(ranks=rank)

        # Retrieve hits
        hits = [hit for data in search_response for hit in data.hits]
        executor = ThreadPoolExecutor(max_workers=10)

        def hit_to_document(hit: resources_pb2.Hit) -> Tuple[Document, float]:
            metadata = json_format.MessageToDict(hit.input.data.metadata)
            h = dict(self._auth.metadata)
            request = requests.get(hit.input.data.text.url, headers=h)

            # override encoding by real educated guess as provided by chardet
            request.encoding = request.apparent_encoding
            requested_text = request.text

            logger.debug(
                f"\tScore {hit.score:.2f} for annotation: {hit.annotation.id}\
                off input: {hit.input.id}, text: {requested_text[:125]}"
            )
            return (Document(page_content=requested_text, metadata=metadata), hit.score)

        # Iterate over hits and retrieve metadata and text
        futures = [executor.submit(hit_to_document, hit) for hit in hits]
        docs_and_scores = [future.result() for future in futures]

        return docs_and_scores

    def similarity_search(
        self,
        query: str,
        k: Optional[int] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Run similarity search using Clarifai.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return.
            If not set, it'll take _number_of_docs. Defaults to None.

        Returns:
            List of Documents most similar to the query and score for each
        """
        docs_and_scores = self.similarity_search_with_score(query, k=k, **kwargs)
        return [doc for doc, _ in docs_and_scores]

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Optional[Embeddings] = None,
        metadatas: Optional[List[dict]] = None,
        user_id: Optional[str] = None,
        app_id: Optional[str] = None,
        number_of_docs: Optional[int] = None,
        pat: Optional[str] = None,
        token: Optional[str] = None,
        **kwargs: Any,
    ) -> Clarifai:
        """Create a Clarifai vectorstore from a list of texts.

        Args:
            user_id (str): User ID.
            app_id (str): App ID.
            texts (List[str]): List of texts to add.
            number_of_docs (Optional[int]): Number of documents
            to return during vector search. Defaults to None.
            pat (Optional[str], optional): Personal access token.
            Defaults to None.
            token (Optional[str], optional): Session token. Defaults to None.
            metadatas (Optional[List[dict]]): Optional list
            of metadatas. Defaults to None.
            kwargs: Additional keyword arguments to be passed to the Search.

        Returns:
            Clarifai: Clarifai vectorstore.
        """
        clarifai_vector_db = cls(
            user_id=user_id,
            app_id=app_id,
            number_of_docs=number_of_docs,
            pat=pat,
            token=token,
            **kwargs,
        )
        clarifai_vector_db.add_texts(texts=texts, metadatas=metadatas)
        return clarifai_vector_db

    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        embedding: Optional[Embeddings] = None,
        user_id: Optional[str] = None,
        app_id: Optional[str] = None,
        number_of_docs: Optional[int] = None,
        pat: Optional[str] = None,
        token: Optional[str] = None,
        **kwargs: Any,
    ) -> Clarifai:
        """Create a Clarifai vectorstore from a list of documents.

        Args:
            user_id (str): User ID.
            app_id (str): App ID.
            documents (List[Document]): List of documents to add.
            number_of_docs (Optional[int]): Number of documents
            to return during vector search. Defaults to None.
            pat (Optional[str], optional): Personal access token. Defaults to None.
            token (Optional[str], optional): Session token. Defaults to None.
            kwargs: Additional keyword arguments to be passed to the Search.

        Returns:
            Clarifai: Clarifai vectorstore.
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return cls.from_texts(
            user_id=user_id,
            app_id=app_id,
            texts=texts,
            number_of_docs=number_of_docs,
            pat=pat,
            metadatas=metadatas,
            token=token,
            **kwargs,
        )
