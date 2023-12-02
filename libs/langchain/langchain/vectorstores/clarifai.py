from __future__ import annotations

import logging
import os
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterable, List, Optional, Tuple

import requests
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

logger = logging.getLogger(__name__)


class Clarifai(VectorStore):
    """`Clarifai AI` vector store.

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
        user_id: Optional[str] = None,
        app_id: Optional[str] = None,
        pat: Optional[str] = None,
        number_of_docs: Optional[int] = None,
        api_base: Optional[str] = None,
    ) -> None:
        """Initialize with Clarifai client.

        Args:
            user_id (Optional[str], optional): User ID. Defaults to None.
            app_id (Optional[str], optional): App ID. Defaults to None.
            pat (Optional[str], optional): Personal access token. Defaults to None.
            number_of_docs (Optional[int], optional): Number of documents to return
            during vector search. Defaults to None.
            api_base (Optional[str], optional): API base. Defaults to None.

        Raises:
            ValueError: If user ID, app ID or personal access token is not provided.
        """
        try:
            from clarifai.auth.helper import DEFAULT_BASE, ClarifaiAuthHelper
            from clarifai.client import create_stub
        except ImportError:
            raise ImportError(
                "Could not import clarifai python package. "
                "Please install it with `pip install clarifai`."
            )

        if api_base is None:
            self._api_base = DEFAULT_BASE

        self._user_id = user_id or os.environ.get("CLARIFAI_USER_ID")
        self._app_id = app_id or os.environ.get("CLARIFAI_APP_ID")
        self._pat = pat or os.environ.get("CLARIFAI_PAT")
        if self._user_id is None or self._app_id is None or self._pat is None:
            raise ValueError(
                "Could not find CLARIFAI_USER_ID, CLARIFAI_APP_ID or\
                CLARIFAI_PAT in your environment. "
                "Please set those env variables with a valid user ID, \
                app ID and personal access token \
                from https://clarifai.com/settings/security."
            )

        self._auth = ClarifaiAuthHelper(
            user_id=self._user_id,
            app_id=self._app_id,
            pat=self._pat,
            base=self._api_base,
        )
        self._stub = create_stub(self._auth)
        self._userDataObject = self._auth.get_user_app_id_proto()
        self._number_of_docs = number_of_docs

    def _post_texts_as_inputs(
        self, texts: List[str], metadatas: Optional[List[dict]] = None
    ) -> List[str]:
        """Post text to Clarifai and return the ID of the input.

        Args:
            text (str): Text to post.
            metadata (dict): Metadata to post.

        Returns:
            str: ID of the input.
        """
        try:
            from clarifai_grpc.grpc.api import resources_pb2, service_pb2
            from clarifai_grpc.grpc.api.status import status_code_pb2
            from google.protobuf.struct_pb2 import Struct  # type: ignore
        except ImportError as e:
            raise ImportError(
                "Could not import clarifai python package. "
                "Please install it with `pip install clarifai`."
            ) from e

        if metadatas is not None:
            assert len(list(texts)) == len(
                metadatas
            ), "Number of texts and metadatas should be the same."

        inputs = []
        for idx, text in enumerate(texts):
            if metadatas is not None:
                input_metadata = Struct()
                input_metadata.update(metadatas[idx])
            inputs.append(
                resources_pb2.Input(
                    data=resources_pb2.Data(
                        text=resources_pb2.Text(raw=text),
                        metadata=input_metadata,
                    )
                )
            )

        post_inputs_response = self._stub.PostInputs(
            service_pb2.PostInputsRequest(
                user_app_id=self._userDataObject,
                inputs=inputs,
            )
        )

        if post_inputs_response.status.code != status_code_pb2.SUCCESS:
            logger.error(post_inputs_response.status)
            raise Exception(
                "Post inputs failed, status: " + post_inputs_response.status.description
            )

        input_ids = []
        for input in post_inputs_response.inputs:
            input_ids.append(input.id)

        return input_ids

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

        Returns:
            List[str]: List of IDs of the added texts.
        """

        ltexts = list(texts)
        length = len(ltexts)
        assert length > 0, "No texts provided to add to the vectorstore."

        if metadatas is not None:
            assert length == len(
                metadatas
            ), "Number of texts and metadatas should be the same."

        batch_size = 32
        input_ids = []
        for idx in range(0, length, batch_size):
            try:
                batch_texts = ltexts[idx : idx + batch_size]
                batch_metadatas = (
                    metadatas[idx : idx + batch_size] if metadatas else None
                )
                result_ids = self._post_texts_as_inputs(batch_texts, batch_metadatas)
                input_ids.extend(result_ids)
                logger.debug(f"Input {result_ids} posted successfully.")
            except Exception as error:
                logger.warning(f"Post inputs failed: {error}")
                traceback.print_exc()

        return input_ids

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
        namespace: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Run similarity search with score using Clarifai.

        Args:
            query (str): Query text to search for.
            k (int): Number of results to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata.
            Defaults to None.

        Returns:
            List[Document]: List of documents most similar to the query text.
        """
        try:
            from clarifai_grpc.grpc.api import resources_pb2, service_pb2
            from clarifai_grpc.grpc.api.status import status_code_pb2
            from google.protobuf import json_format  # type: ignore
            from google.protobuf.struct_pb2 import Struct  # type: ignore
        except ImportError as e:
            raise ImportError(
                "Could not import clarifai python package. "
                "Please install it with `pip install clarifai`."
            ) from e

        # Get number of docs to return
        if self._number_of_docs is not None:
            k = self._number_of_docs

        req = service_pb2.PostAnnotationsSearchesRequest(
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
            pagination=service_pb2.Pagination(page=1, per_page=k),
        )

        # Add filter by metadata if provided.
        if filter is not None:
            search_metadata = Struct()
            search_metadata.update(filter)
            f = req.searches[0].query.filters.add()
            f.annotation.data.metadata.update(search_metadata)

        post_annotations_searches_response = self._stub.PostAnnotationsSearches(req)

        # Check if search was successful
        if post_annotations_searches_response.status.code != status_code_pb2.SUCCESS:
            raise Exception(
                "Post searches failed, status: "
                + post_annotations_searches_response.status.description
            )

        # Retrieve hits
        hits = post_annotations_searches_response.hits

        executor = ThreadPoolExecutor(max_workers=10)

        def hit_to_document(hit: resources_pb2.Hit) -> Tuple[Document, float]:
            metadata = json_format.MessageToDict(hit.input.data.metadata)
            h = {"Authorization": f"Key {self._auth.pat}"}
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
        k: int = 4,
        **kwargs: Any,
    ) -> List[Document]:
        """Run similarity search using Clarifai.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.

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
        user_id: Optional[str] = None,
        app_id: Optional[str] = None,
        pat: Optional[str] = None,
        number_of_docs: Optional[int] = None,
        api_base: Optional[str] = None,
        **kwargs: Any,
    ) -> Clarifai:
        """Create a Clarifai vectorstore from a list of texts.

        Args:
            user_id (str): User ID.
            app_id (str): App ID.
            texts (List[str]): List of texts to add.
            pat (Optional[str]): Personal access token. Defaults to None.
            number_of_docs (Optional[int]): Number of documents to return
            during vector search. Defaults to None.
            api_base (Optional[str]): API base. Defaults to None.
            metadatas (Optional[List[dict]]): Optional list of metadatas.
            Defaults to None.

        Returns:
            Clarifai: Clarifai vectorstore.
        """
        clarifai_vector_db = cls(
            user_id=user_id,
            app_id=app_id,
            pat=pat,
            number_of_docs=number_of_docs,
            api_base=api_base,
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
        pat: Optional[str] = None,
        number_of_docs: Optional[int] = None,
        api_base: Optional[str] = None,
        **kwargs: Any,
    ) -> Clarifai:
        """Create a Clarifai vectorstore from a list of documents.

        Args:
            user_id (str): User ID.
            app_id (str): App ID.
            documents (List[Document]): List of documents to add.
            pat (Optional[str]): Personal access token. Defaults to None.
            number_of_docs (Optional[int]): Number of documents to return
            during vector search. Defaults to None.
            api_base (Optional[str]): API base. Defaults to None.

        Returns:
            Clarifai: Clarifai vectorstore.
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return cls.from_texts(
            user_id=user_id,
            app_id=app_id,
            texts=texts,
            pat=pat,
            number_of_docs=number_of_docs,
            api_base=api_base,
            metadatas=metadatas,
        )
