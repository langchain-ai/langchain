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
            from clarifai.auth.helper import DEFAULT_BASE, ClarifaiAuthHelper
            from clarifai.client import create_stub
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
            pat = os.environ["CLARIFAI_PAT_KEY"]

        if api_base is None:
            api_base = DEFAULT_BASE

        auth = ClarifaiAuthHelper(user_id=user_id, app_id=app_id, pat=pat, base=api_base)
        stub = create_stub(auth)
        userDataObject = auth.get_user_app_id_proto()

        self._stub = stub
        self._auth = auth
        self._userDataObject = userDataObject
        self._number_of_docs = number_of_docs

    def _post_text_input(self, text: str, metadata: dict) -> str:
        """Post text to Clarifai and return the ID of the input."""
        from clarifai_grpc.grpc.api import resources_pb2, service_pb2
        from clarifai_grpc.grpc.api.status import status_code_pb2
        from google.protobuf.struct_pb2 import Struct

        input_metadata = Struct()
        input_metadata.update(metadata)

        post_inputs_response = self._stub.PostInputs(
            service_pb2.PostInputsRequest(
                user_app_id=self._userDataObject,
                inputs=[
                    resources_pb2.Input(
                        data=resources_pb2.Data(
                            text=resources_pb2.Text(raw=text),
                            metadata=input_metadata,
                        )
                    )
                ],
            )
        )

        if post_inputs_response.status.code != status_code_pb2.SUCCESS:
            raise Exception("Post inputs failed, status: " + post_inputs_response.status.description)

        input_id = post_inputs_response.inputs[0].id

        return input_id

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add texts to the Clarifai vectorstore. This will push the text to a Clarifai application.
        Application use base workflow that create and store embedding for each text.
        Make sure you are using a base workflow that is compatible with text (such as Language Understanding).

        Args:
            texts (Iterable[str]): Texts to add to the vectorstore.
            metadatas (Optional[List[dict]], optional): Optional list of metadatas.
            ids (Optional[List[str]], optional): Optional list of IDs.

        Returns:
            List[str]: List of IDs of the added texts.
        """

        assert len(texts) > 0, "No texts provided to add to the vectorstore."
        assert len(texts) == len(metadatas), "Number of texts and metadatas should be the same."

        input_ids = []
        for idx, text in enumerate(texts):
            try:
                metadata = metadatas[i] if metadatas else {}
                input_id = self._post_text_input(text, metadata)
                input_ids.append(input_id)
                print(f"Input {input_id} posted successfully.")
            except Exception as error:
                print(f"Post inputs failed: {error}" % error)
                traceback.print_exc()

        return input_ids

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
                pagination=service_pb2.Pagination(page=1, per_page=k),
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
        user_id: str,
        app_id: str,
        texts: List[str],
        pat: Optional[str] = None,
        number_of_docs: Optional[int] = None,
        api_base: Optional[str] = None,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> Clarifai:
        """Create a Clarifai vectorstore from a list of texts.

        Args:
            user_id (str): User ID.
            app_id (str): App ID.
            texts (List[str]): List of texts to add.
            pat (Optional[str]): Personal access token. Defaults to None.
            number_of_docs (Optional[int]): Number of documents to return during vector search. Defaults to None.
            api_base (Optional[str]): API base. Defaults to None.
            metadatas (Optional[List[dict]]): Optional list of metadatas. Defaults to None.
            ids (Optional[List[str]]): List of document IDs. Defaults to None.

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
        user_id: str,
        app_id: str,
        documents: List[Document],
        pat: Optional[str] = None,
        number_of_docs: Optional[int] = None,
        api_base: Optional[str] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Clarifai:
        """Create a Clarifai vectorstore from a list of documents.

        Args:
            user_id (str): User ID.
            app_id (str): App ID.
            documents (List[Document]): List of documents to add.
            pat (Optional[str]): Personal access token. Defaults to None.
            number_of_docs (Optional[int]): Number of documents to return during vector search. Defaults to None.
            api_base (Optional[str]): API base. Defaults to None.
            ids (Optional[List[str]]): List of document IDs. Defaults to None.

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
