from __future__ import annotations

import os
import time
import uuid
from typing import Any, Dict, List, Optional, Type

import numpy as np

from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from langchain.vectorstores.base import VectorStore

DEFAULT_TOPN = 4


class Vearch(VectorStore):
    _DEFAULT_TABLE_NAME = "langchain_vearch"

    def __init__(
        self,
        embedding_function: Embeddings,
        table_name: str = _DEFAULT_TABLE_NAME,
        metadata_path: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize vearch vector store"""
        try:
            import vearch
        except ImportError:
            raise ImportError(
                "Could not import vearch python package. "
                "Please install it with `pip install vearch`."
            )

        if metadata_path is None:
            metadata_path = os.getcwd().replace("\\", "/")
        if not os.path.isdir(metadata_path):
            os.makedirs(metadata_path)
        log_path = os.path.join(metadata_path, "log")
        if not os.path.isdir(log_path):
            os.makedirs(log_path)
        self.vearch_engine = vearch.Engine(metadata_path, log_path)

        if not table_name:
            table_name = self._DEFAULT_TABLE_NAME
            table_name += "_"
            table_name += str(uuid.uuid4()).split("-")[-1]
        self.using_table_name = table_name
        self.using_metapath = metadata_path
        self.embedding_func = embedding_function

    @property
    def embeddings(self) -> Optional[Embeddings]:
        return self.embedding_func

    @classmethod
    def from_documents(
        cls: Type[Vearch],
        documents: List[Document],
        embedding: Embeddings,
        table_name: str = "langchain_vearch",
        metadata_path: Optional[str] = None,
        **kwargs: Any,
    ) -> Vearch:
        """Return Vearch VectorStore"""

        texts = [d.page_content for d in documents]
        metadatas = [d.metadata for d in documents]

        return cls.from_texts(
            texts=texts,
            embedding=embedding,
            metadatas=metadatas,
            table_name=table_name,
            metadata_path=metadata_path,
            **kwargs,
        )

    @classmethod
    def from_texts(
        cls: Type[Vearch],
        texts: List[str],
        embedding: Embeddings,
        metadatas: List[dict],
        table_name: str = _DEFAULT_TABLE_NAME,
        metadata_path: Optional[str] = None,
        **kwargs: Any,
    ) -> Vearch:
        """Return Vearch VectorStore"""

        vearch_db = cls(
            embedding_function=embedding,
            table_name=table_name,
            metadata_path=metadata_path,
        )
        vearch_db.add_texts(texts=texts, metadatas=metadatas)
        return vearch_db

    def _create_table(
        self,
        dim: int = 1024,
        filed_list: List[dict] = [
            {"filed": "text", "type": "str"},
            {"filed": "metadata", "type": "str"},
        ],
    ) -> int:
        """Create VectorStore Table

        Args:
            dim:dimension of vector
            fileds_list: the filed you want to store

        Return:
            code,0 for success,1 for failed
        """
        import vearch
        from vearch import GammaFieldInfo, GammaVectorInfo

        type_dict = {"int": vearch.dataType.INT, "str": vearch.dataType.STRING}
        engine_info = {
            "index_size": 10000,
            "retrieval_type": "IVFPQ",
            "retrieval_param": {"ncentroids": 2048, "nsubvector": 32},
        }
        fields = [
            GammaFieldInfo(fi["filed"], type_dict[fi["type"]]) for fi in filed_list
        ]
        vector_field = GammaVectorInfo(
            name="text_embedding",
            type=vearch.dataType.VECTOR,
            is_index=True,
            dimension=dim,
            model_id="",
            store_type="MemoryOnly",
            store_param={"cache_size": 10000},
            has_source=False,
        )
        response_code = self.vearch_engine.create_table(
            engine_info,
            name=self.using_table_name,
            fields=fields,
            vector_field=vector_field,
        )
        return response_code

    def add_texts(
        self,
        texts: List[str],
        metadatas: List[dict],
        **kwargs: Any,
    ) -> List[str]:
        """
        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        embeddings = None
        if self.embedding_func is not None:
            embeddings = self.embedding_func.embed_documents(list(texts))
        table_path = os.path.join(
            self.using_metapath, self.using_table_name + ".schema"
        )
        if not os.path.exists(table_path):
            if embeddings is None:
                raise ValueError("embeddings is None")
            dim = len(embeddings[0])
            response_code = self._create_table(dim)
            if response_code:
                raise ValueError("create table failed!!!")

        if embeddings is not None:
            doc_items = []
            for i in range(len(embeddings)):
                profiles = {}
                profiles["text"] = texts[i]
                profiles["metadata"] = metadatas[i]["source"]
                profiles["text_embedding"] = embeddings[i][:] / (
                    np.linalg.norm(embeddings[i][:])
                )
                doc_items.append(profiles)
            docid = self.vearch_engine.add(doc_items)
            t_time = 0
            while len(docid) != len(embeddings):
                time.sleep(0.5)
                if t_time > 6:
                    break
                t_time += 1
            self.vearch_engine.dump()

        return docid

    def _load(self) -> None:
        """
        load vearch engine
        """
        self.vearch_engine.load()

    @classmethod
    def load_local(
        cls,
        embedding: Embeddings,
        table_name: str = _DEFAULT_TABLE_NAME,
        metadata_path: Optional[str] = None,
        **kwargs: Any,
    ) -> Vearch:
        """Load the local specified table.
        Returns:
            Success or failure of loading the local specified table
        """
        if not metadata_path:
            raise ValueError("No metadata path!!!")
        if not table_name:
            raise ValueError("No table name!!!")
        table_path = os.path.join(metadata_path, table_name + ".schema")
        if not os.path.exists(table_path):
            raise ValueError("vearch vectorbase table not exist!!!")
        vearch_db = cls(
            embedding_function=embedding,
            table_name=table_name,
            metadata_path=metadata_path,
        )
        vearch_db._load()
        return vearch_db

    def similarity_search(
        self,
        query: str,
        k: int = DEFAULT_TOPN,
        **kwargs: Any,
    ) -> List[Document]:
        """
        Return docs most similar to query.

        """
        if self.vearch_engine is None:
            raise ValueError("Vearch engine is None!!!")
        if self.embedding_func is None:
            raise ValueError("embedding_func is None!!!")
        embeddings = self.embedding_func.embed_query(query)
        docs = self.similarity_search_with_score_by_vector(embeddings, k)
        return docs

    def similarity_search_with_score_by_vector(
        self,
        embeddings: List[float],
        k: int = DEFAULT_TOPN,
        min_score: float = 0.0,
        **kwargs: Any,
    ) -> List[Document]:
        """The most k similar documents and scores of the specified query.

        Args:
            embeddings: embedding vector of the query.
            k: The k most similar documents to the text query.
            min_score: the score of similar documents to the text query
        Returns:
            The k most similar documents to the specified text query.
            0 is dissimilar, 1 is the most similar.
        """
        query_data = {
            "vector": [
                {
                    "field": "text_embedding",
                    "feature": embeddings / (np.linalg.norm(embeddings[:])),
                    "min_score": min_score,
                }
            ],
            "fields": [],
            "is_brute_search": 1,
            "retrieval_param": {"metric_type": "InnerProduct", "nprobe": 20},
            "topn": k,
        }
        query_result = self.vearch_engine.search(query_data)
        docs = []
        for item in query_result[0]["result_items"]:
            content = ""
            meta_data = {}
            for item_key in item:
                if item_key == "text":
                    content = item[item_key]
                    continue
                if item_key == "metadata":
                    meta_data["source"] = item[item_key]
                    continue
            docs.append(Document(page_content=content, metadata=meta_data))
        return docs

    def delete(
        self,
        ids: List[str],
        **kwargs: Any,
    ) -> Optional[bool]:
        """Delete the documents which have the specified ids.

        Args:
            ids: The ids of the embedding vectors.
            **kwargs: Other keyword arguments that subclasses might use.
        Returns:
            Optional[bool]: True if deletion is successful.
            False otherwise, None if not implemented.
        """
        if self.vearch_engine is None:
            raise ValueError("Verach Engine is None!!!")
        ret: Optional[bool] = None
        tmp_res = []
        if ids is None or ids.__len__() == 0:
            return ret
        for _id in ids:
            ret = self.vearch_engine.del_doc(_id)
            tmp_res.append(ret)
        ret = all(i == 0 for i in tmp_res)
        return ret

    def get(
        self,
        ids: List[str],
        **kwargs: Any,
    ) -> Dict[str, Document]:
        """Return docs according ids.

        Args:
            ids: The ids of the embedding vectors.
        Returns:
            Documents which satisfy the input conditions.
        """

        if self.vearch_engine is None:
            raise ValueError("vearch engine is None!!!")
        results: Dict[str, Document] = {}
        for id in ids:
            docs_detail = self.vearch_engine.get_doc_by_id(id)
            if docs_detail == {}:
                continue

            content = ""
            meta_info = {}
            for field in docs_detail:
                if field == "text":
                    content = docs_detail[field]
                    continue
                elif field == "metadata":
                    meta_info["source"] = docs_detail[field]
                    continue
            results[docs_detail["_id"]] = Document(
                page_content=content, metadata=meta_info
            )
        return results
