"""Wrapper around Atlas by Nomic"""
from __future__ import annotations

import uuid
import logging
from typing import Any, Dict, Iterable, List, Optional

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.base import VectorStore

logger = logging.getLogger()


class AtlasDB(VectorStore):
    """Wrapper around Atlas: Nomic's neural database and rhizomatic instrument

    To use, you should have the ``nomic`` python package installed.

    Example:
        .. code-block:: python

                from langchain.vectorstores import AtlasDB
                from langchain.embeddings.openai import OpenAIEmbeddings

                embeddings = OpenAIEmbeddings()
                vectorstore = AtlasDB("my_project", embeddings.embed_query)
    """
    _ATLAS_DEFAULT_ID_FIELD = "_id"

    def __init__(
        self,
        name: str,
        embedding_function: Optional[Embeddings] = None,
        api_key: Optional[str] = None,
        description: str = 'A description for your project',
        is_public: bool = True,
        reset_project_if_exists = False,
    ) -> None:
        """
        Initialize the Atlas Client

        Args:
            name (str): The name of your project. If the project already exists, it will be loaded.
            embedding_function (Optional[Callable]): An optional function used for embedding your data. If None, data will be embedded with Nomic's embed model.
            api_key (str): Your nomic API key
            description (str): A description for your project.
            is_public (bool): Whether your project is publicly accessible. True by default.
            reset_project_if_exists (bool): Whether to reset this project if it already exists. Default False. Generally userful during development and testing.
        """
        try:
            import nomic
            from nomic import atlas, AtlasProject, AtlasProjection
        except ImportError:
            raise ValueError(
                "Could not import nomic python package. "
                "Please it install it with `pip install nomic`."
            )

        if api_key is None:
            raise ValueError("No API key provided. Sign up at atlas.nomic.ai!")
        nomic.login(api_key)
        self._embedding_function = embedding_function

        # Check if the project exists, create it if not
        self.project = AtlasProject(name=name,
                                    description=description,
                                    is_public=is_public,
                                    reset_project_if_exists=reset_project_if_exists,
                                    unique_id_field=_ATLAS_DEFAULT_ID_FIELD)

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        refresh=True,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts (Iterable[str]): Texts to add to the vectorstore.
            metadatas (Optional[List[dict]], optional): Optional list of metadatas.
            ids (Optional[List[str]]): An optional list of ids.
            refresh(bool): Whether or not to refresh indices with the updated data. Default True.
        Returns:
            List[str]: List of IDs of the added texts.
        """

        if 'text' in metadatas[0].keys():
            raise ValueError('Cannot accept key text in metadata!')

        texts = list(texts)
        if ids is None:
            ids = [uuid.uuid1() for _ in texts]

        #Embedding upload case
        if self._embedding_function is not None:
            embeddings = self._embedding_function.embed_documents(texts)
            if metadatas is None:
                data = [{_ATLAS_DEFAULT_ID_FIELD: ids[i]} for i, _ in enumerate(texts)]
            else:
                for i in range(len(metadatas)):
                    metadatas[i][_ATLAS_DEFAULT_ID_FIELD] = ids[i]
                data = metadats

            self.project._validate_map_data_inputs([],
                                                   id_field=_ATLAS_DEFAULT_ID_FIELD,
                                                   data=data)
            with self.project.wait_for_project_lock():
                self.project.add_embeddings(embeddings=embeddings,
                                             data=data)
        #Text upload case
        else:
            if metadatas is None:
                data = [{'text': text,
                         _ATLAS_DEFAULT_ID_FIELD: ids[i]}
                        for i, text in enumerate(texts)]
            else:
                for i, text in enumerate(texts):
                    metadatas[i]['text'] = texts
                    metadatas[i][_ATLAS_DEFAULT_ID_FIELD] = ids[i]
                data = metadatas

            self.project._validate_map_data_inputs([],
                                                   id_field=_ATLAS_DEFAULT_ID_FIELD,
                                                   data=data)

            with self.project.wait_for_project_lock():
                self.project.add_text(data)

        if refresh:
            if len(self.project.indices()) > 0:
                with self.project.wait_for_project_lock():
                    self.project.rebuild_maps()

        return ids

    def create_index(self, **kwargs) -> AtlasProjection:
        '''
        Creates an index in your project.
        See https://docs.nomic.ai/atlas_api.html#nomic.project.AtlasProject.create_index for full detail
        '''
        with self.project.wait_for_project_lock():
            return self.project.create_index(**kwargs)

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Document]:
        """Run similarity search with AtlasDB

        Args:
            query (str): Query text to search for.
            k (int): Number of results to return. Defaults to 4.

        Returns:
            List[Document]: List of documents most similar to the query text.
        """
        if self._embedding_function is None:
            raise NotImplementedError('AtlasDB requires an embedding_function for text similarity search!')

        embedding = self._embedding_function.embed_documents([query])[0]
        with self.project.wait_for_project_lock():
            neighbors, _ = self.project.vector_search(queries=embedding, k=k)
            datas = self.project.get_data(ids=neighbors[0])

        docs = [
            Document(page_content=datas[i]['text'], metadata=datas[i])
            for i, neighbor in enumerate(neighbors)
        ]
        return docs

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        name: str,
        api_key: str,
        embedding: Optional[Embeddings] = None,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        description: str = 'A description for your project',
        is_public: bool = True,
        reset_project_if_exists=False,
        index_kwargs=None,
        **kwargs: Any,
    ) -> AtlasDB:
        """Create an AtlasDB vectorstore from a raw documents.

        Args:
            texts (List[str]): The list of texts to ingest.
            name (str): Name of the project to create.
            api_key (str): Your nomic API key,
            embedding (Optional[Embeddings]): Embedding function. Defaults to None.
            metadatas (Optional[List[dict]]): List of metadatas. Defaults to None.
            ids (Optional[List[str]]): Optional list of document IDs. If None, ids will be auto created
            description (str): A description for your project.
            is_public (bool): Whether your project is publicly accessible. True by default.
            reset_project_if_exists (bool): Whether to reset this project if it already exists. Default False. Generally userful during development and testing.
            index_kwargs (Optional[dict]): Dict of kwargs for index creation. See https://docs.nomic.ai/atlas_api.html#nomic.project.AtlasProject.create_index

        Returns:
            AtlasDB: Nomic's neural database and finest rhizomatic instrument
        """

        #Inject relevant kwargs
        all_index_kwargs = {'name': name + '_index',
                            'indexed_field': 'text'}
        if index_kwargs is not None:
            for k, v in index_kwargs.items():
                all_index_kwargs[k] = v

        #Build project
        project = cls(
            name,
            embedding_function=embedding,
            api_key=api_key,
            description='A description for your project',
            is_public=is_public,
            reset_project_if_exists=reset_project_if_exists,
        )
        with self.project.wait_for_project_lock():
            project.add_texts(texts=texts, metadatas=metadatas, ids=ids)
            project.create_index(**index_kwargs)
        return project

    @classmethod
    def from_documents(
        cls,
        name: str,
        api_key: str,
        documents: List[Document],
        embedding: Optional[Embeddings] = None,
        ids: Optional[List[str]] = None,
        persist_directory: Optional[str] = None,
        description: str = 'A description for your project',
        is_public: bool = True,
        reset_project_if_exists=False,
        index_kwargs=None,
        **kwargs: Any,
    ) -> AtlasDB:
        """Create an AtlasDB vectorstore from a list of documents.

        Args:
            name (str): Name of the collection to create.
            api_key (str): Your nomic API key,
            documents (List[Document]): List of documents to add to the vectorstore.
            embedding (Optional[Embeddings]): Embedding function. Defaults to None.
            ids (Optional[List[str]]): Optional list of document IDs. If None, ids will be auto created
            description (str): A description for your project.
            is_public (bool): Whether your project is publicly accessible. True by default.
            reset_project_if_exists (bool): Whether to reset this project if it already exists. Default False. Generally userful during development and testing.
            index_kwargs (Optional[dict]): Dict of kwargs for index creation. See https://docs.nomic.ai/atlas_api.html#nomic.project.AtlasProject.create_index

        Returns:
            AtlasDB: Nomic's neural database and finest rhizomatic instrument
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return cls.from_texts(
            name=name,
            api_key=api_key,
            texts=texts,
            embedding=embedding,
            metadatas=metadatas,
            ids=ids,
            description=description,
            is_public=is_public,
            reset_project_if_exists=reset_project_if_exists,
            index_kwargs=index_kwargs,
        )
