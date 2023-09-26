from __future__ import annotations

import os
import time
import uuid
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple, Type

import numpy as np

from langchain.docstore.document import Document
from langchain.schema.embeddings import Embeddings
from langchain.schema.vectorstore import VectorStore

if TYPE_CHECKING:
    import vearch

DEFAULT_TOPN = 4


class Vearch(VectorStore):
    _DEFAULT_TABLE_NAME = "langchain_vearch"
    _DEFAULT_CLUSTER_DB_NAME = "cluster_client_db"
    _DEFAULT_VERSION = 1

    def __init__(
        self,
        embedding_function: Embeddings,
        path_or_url: Optional[str] = None,
        table_name: str = _DEFAULT_TABLE_NAME,
        db_name: str = _DEFAULT_CLUSTER_DB_NAME,
        flag: int = _DEFAULT_VERSION,
        **kwargs: Any,
    ) -> None:
        """Initialize vearch vector store
        flag 1 for cluster,0 for standalone
        """
        try:
            if flag:
                import vearch_cluster
            else:
                import vearch
        except ImportError:
            raise ValueError(
                "Could not import suitable python package. "
                "Please install it with `pip install vearch or vearch_cluster`."
            )

        if flag:
            if path_or_url is None:
                raise ValueError("Please input url of cluster")
            if not db_name:
                db_name = self._DEFAULT_CLUSTER_DB_NAME
                db_name += "_"
                db_name += str(uuid.uuid4()).split("-")[-1]
            self.using_db_name = db_name
            self.url = path_or_url
            self.vearch = vearch_cluster.VearchCluster(path_or_url)

        else:
            if path_or_url is None:
                metadata_path = os.getcwd().replace("\\", "/")
            else:
                metadata_path = path_or_url
            if not os.path.isdir(metadata_path):
                os.makedirs(metadata_path)
            log_path = os.path.join(metadata_path, "log")
            if not os.path.isdir(log_path):
                os.makedirs(log_path)
            self.vearch = vearch.Engine(metadata_path, log_path)
            self.using_metapath = metadata_path
        if not table_name:
            table_name = self._DEFAULT_TABLE_NAME
            table_name += "_"
            table_name += str(uuid.uuid4()).split("-")[-1]
        self.using_table_name = table_name
        self.embedding_func = embedding_function
        self.flag = flag

    @property
    def embeddings(self) -> Optional[Embeddings]:
        return self.embedding_func

    @classmethod
    def from_documents(
        cls: Type[Vearch],
        documents: List[Document],
        embedding: Embeddings,
        path_or_url: Optional[str] = None,
        table_name: str = _DEFAULT_TABLE_NAME,
        db_name: str = _DEFAULT_CLUSTER_DB_NAME,
        flag: int = _DEFAULT_VERSION,
        **kwargs: Any,
    ) -> Vearch:
        """Return Vearch VectorStore"""

        texts = [d.page_content for d in documents]
        metadatas = [d.metadata for d in documents]

        return cls.from_texts(
            texts=texts,
            embedding=embedding,
            metadatas=metadatas,
            path_or_url=path_or_url,
            table_name=table_name,
            db_name=db_name,
            flag=flag,
            **kwargs,
        )

    @classmethod
    def from_texts(
        cls: Type[Vearch],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        path_or_url: Optional[str] = None,
        table_name: str = _DEFAULT_TABLE_NAME,
        db_name: str = _DEFAULT_CLUSTER_DB_NAME,
        flag: int = _DEFAULT_VERSION,
        **kwargs: Any,
    ) -> Vearch:
        """Return Vearch VectorStore"""

        vearch_db = cls(
            embedding_function=embedding,
            embedding=embedding,
            path_or_url=path_or_url,
            db_name=db_name,
            table_name=table_name,
            flag=flag,
        )
        vearch_db.add_texts(texts=texts, metadatas=metadatas)
        return vearch_db

    def _create_table(
        self,
        dim: int = 1024,
        field_list: List[dict] = [
            {"field": "text", "type": "str"},
            {"field": "metadata", "type": "str"},
        ],
    ) -> int:
        """
        Create VectorStore Table
        Args:
            dim:dimension of vector
            fields_list: the field you want to store
        Return:
            code,0 for success,1 for failed
        """

        type_dict = {"int": vearch.dataType.INT, "str": vearch.dataType.STRING}
        engine_info = {
            "index_size": 10000,
            "retrieval_type": "IVFPQ",
            "retrieval_param": {"ncentroids": 2048, "nsubvector": 32},
        }
        fields = [
            vearch.GammaFieldInfo(fi["field"], type_dict[fi["type"]])
            for fi in field_list
        ]
        vector_field = vearch.GammaVectorInfo(
            name="text_embedding",
            type=vearch.dataType.VECTOR,
            is_index=True,
            dimension=dim,
            model_id="",
            store_type="MemoryOnly",
            store_param={"cache_size": 10000},
            has_source=False,
        )
        response_code = self.vearch.create_table(
            engine_info,
            name=self.using_table_name,
            fields=fields,
            vector_field=vector_field,
        )
        return response_code

    def _create_space(
        self,
        dim: int = 1024,
    ) -> int:
        """
        Create VectorStore space
        Args:
            dim:dimension of vector
        Return:
            code,0 failed for ,1 for success
        """
        space_config = {
            "name": self.using_table_name,
            "partition_num": 1,
            "replica_num": 1,
            "engine": {
                "name": "gamma",
                "index_size": 1,
                "retrieval_type": "FLAT",
                "retrieval_param": {
                    "metric_type": "L2",
                },
            },
            "properties": {
                "text": {
                    "type": "string",
                },
                "metadata": {
                    "type": "string",
                },
                "text_embedding": {
                    "type": "vector",
                    "index": True,
                    "dimension": dim,
                    "store_type": "MemoryOnly",
                },
            },
        }
        response_code = self.vearch.create_space(self.using_db_name, space_config)

        return response_code

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """
        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        embeddings = None
        if self.embedding_func is not None:
            embeddings = self.embedding_func.embed_documents(list(texts))
        if embeddings is None:
            raise ValueError("embeddings is None")
        if self.flag:
            dbs_list = self.vearch.list_dbs()
            if self.using_db_name not in dbs_list:
                create_db_code = self.vearch.create_db(self.using_db_name)
                if not create_db_code:
                    raise ValueError("create db failed!!!")
            space_list = self.vearch.list_spaces(self.using_db_name)
            if self.using_table_name not in space_list:
                create_space_code = self._create_space(len(embeddings[0]))
                if not create_space_code:
                    raise ValueError("create space failed!!!")
            docid = []
            if embeddings is not None and metadatas is not None:
                for text, metadata, embed in zip(texts, metadatas, embeddings):
                    profiles: dict[str, Any] = {}
                    profiles["text"] = text
                    profiles["metadata"] = metadata["source"]
                    embed_np = np.array(embed)
                    profiles["text_embedding"] = {
                        "feature": (embed_np / np.linalg.norm(embed_np)).tolist()
                    }
                    insert_res = self.vearch.insert_one(
                        self.using_db_name, self.using_table_name, profiles
                    )
                    if insert_res["status"] == 200:
                        docid.append(insert_res["_id"])
                        continue
                    else:
                        retry_insert = self.vearch.insert_one(
                            self.using_db_name, self.using_table_name, profiles
                        )
                        docid.append(retry_insert["_id"])
                        continue
        else:
            table_path = os.path.join(
                self.using_metapath, self.using_table_name + ".schema"
            )
            if not os.path.exists(table_path):
                dim = len(embeddings[0])
                response_code = self._create_table(dim)
                if response_code:
                    raise ValueError("create table failed!!!")
            if embeddings is not None and metadatas is not None:
                doc_items = []
                for text, metadata, embed in zip(texts, metadatas, embeddings):
                    profiles_v: dict[str, Any] = {}
                    profiles_v["text"] = text
                    profiles_v["metadata"] = metadata["source"]
                    embed_np = np.array(embed)
                    profiles_v["text_embedding"] = embed_np / np.linalg.norm(embed_np)
                    doc_items.append(profiles_v)

                docid = self.vearch.add(doc_items)
                t_time = 0
                while len(docid) != len(embeddings):
                    time.sleep(0.5)
                    if t_time > 6:
                        break
                    t_time += 1
                self.vearch.dump()
        return docid

    def _load(self) -> None:
        """
        load vearch engine for standalone vearch
        """
        self.vearch.load()

    @classmethod
    def load_local(
        cls,
        embedding: Embeddings,
        path_or_url: Optional[str] = None,
        table_name: str = _DEFAULT_TABLE_NAME,
        db_name: str = _DEFAULT_CLUSTER_DB_NAME,
        flag: int = _DEFAULT_VERSION,
        **kwargs: Any,
    ) -> Vearch:
        """Load the local specified table of standalone vearch.
        Returns:
            Success or failure of loading the local specified table
        """
        if not path_or_url:
            raise ValueError("No metadata path!!!")
        if not table_name:
            raise ValueError("No table name!!!")
        table_path = os.path.join(path_or_url, table_name + ".schema")
        if not os.path.exists(table_path):
            raise ValueError("vearch vectorbase table not exist!!!")

        vearch_db = cls(
            embedding_function=embedding,
            path_or_url=path_or_url,
            table_name=table_name,
            db_name=db_name,
            flag=flag,
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
        if self.embedding_func is None:
            raise ValueError("embedding_func is None!!!")
        embeddings = self.embedding_func.embed_query(query)
        docs = self.similarity_search_by_vector(embeddings, k)
        return docs

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = DEFAULT_TOPN,
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
        embed = np.array(embedding)
        if self.flag:
            query_data = {
                "query": {
                    "sum": [
                        {
                            "field": "text_embedding",
                            "feature": (embed / np.linalg.norm(embed)).tolist(),
                        }
                    ],
                },
                "size": k,
                "fields": ["text", "metadata"],
            }
            query_result = self.vearch.search(
                self.using_db_name, self.using_table_name, query_data
            )
            res = query_result["hits"]["hits"]
        else:
            query_data = {
                "vector": [
                    {
                        "field": "text_embedding",
                        "feature": embed / np.linalg.norm(embed),
                    }
                ],
                "fields": [],
                "is_brute_search": 1,
                "retrieval_param": {"metric_type": "InnerProduct", "nprobe": 20},
                "topn": k,
            }
            query_result = self.vearch.search(query_data)
            res = query_result[0]["result_items"]
        docs = []
        for item in res:
            content = ""
            meta_data = {}
            if self.flag:
                item = item["_source"]
            for item_key in item:
                if item_key == "text":
                    content = item[item_key]
                    continue
                if item_key == "metadata":
                    meta_data["source"] = item[item_key]
                    continue
            docs.append(Document(page_content=content, metadata=meta_data))
        return docs

    def similarity_search_with_score(
        self,
        query: str,
        k: int = DEFAULT_TOPN,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """The most k similar documents and scores of the specified query.
        Args:
            embeddings: embedding vector of the query.
            k: The k most similar documents to the text query.
            min_score: the score of similar documents to the text query
        Returns:
            The k most similar documents to the specified text query.
            0 is dissimilar, 1 is the most similar.
        """
        if self.embedding_func is None:
            raise ValueError("embedding_func is None!!!")
        embeddings = self.embedding_func.embed_query(query)
        embed = np.array(embeddings)
        if self.flag:
            query_data = {
                "query": {
                    "sum": [
                        {
                            "field": "text_embedding",
                            "feature": (embed / np.linalg.norm(embed)).tolist(),
                        }
                    ],
                },
                "size": k,
                "fields": ["text_embedding", "text", "metadata"],
            }
            query_result = self.vearch.search(
                self.using_db_name, self.using_table_name, query_data
            )
            res = query_result["hits"]["hits"]
        else:
            query_data = {
                "vector": [
                    {
                        "field": "text_embedding",
                        "feature": embed / np.linalg.norm(embed),
                    }
                ],
                "fields": [],
                "is_brute_search": 1,
                "retrieval_param": {"metric_type": "InnerProduct", "nprobe": 20},
                "topn": k,
            }
            query_result = self.vearch.search(query_data)
            res = query_result[0]["result_items"]
        results: List[Tuple[Document, float]] = []
        for item in res:
            content = ""
            meta_data = {}
            if self.flag:
                score = item["_score"]
                item = item["_source"]
            for item_key in item:
                if item_key == "text":
                    content = item[item_key]
                    continue
                if item_key == "metadata":
                    meta_data["source"] = item[item_key]
                    continue
                if self.flag != 1 and item_key == "score":
                    score = item[item_key]
                    continue
            tmp_res = (Document(page_content=content, metadata=meta_data), score)
            results.append(tmp_res)
        return results

    def _similarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        return self.similarity_search_with_score(query, k, **kwargs)

    def delete(
        self,
        ids: Optional[List[str]] = None,
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

        ret: Optional[bool] = None
        tmp_res = []
        if ids is None or ids.__len__() == 0:
            return ret
        for _id in ids:
            if self.flag:
                ret = self.vearch.delete(self.using_db_name, self.using_table_name, _id)
            else:
                ret = self.vearch.del_doc(_id)
            tmp_res.append(ret)
        ret = all(i == 0 for i in tmp_res)
        return ret

    def get(
        self,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Document]:
        """Return docs according ids.

        Args:
            ids: The ids of the embedding vectors.
        Returns:
            Documents which satisfy the input conditions.
        """

        results: Dict[str, Document] = {}
        if ids is None or ids.__len__() == 0:
            return results
        if self.flag:
            query_data = {"query": {"ids": ids}}
            docs_detail = self.vearch.mget_by_ids(
                self.using_db_name, self.using_table_name, query_data
            )
            for record in docs_detail:
                if record["found"] is False:
                    continue
                content = ""
                meta_info = {}
                for field in record["_source"]:
                    if field == "text":
                        content = record["_source"][field]
                        continue
                    elif field == "metadata":
                        meta_info["source"] = record["_source"][field]
                        continue
                results[record["_id"]] = Document(
                    page_content=content, metadata=meta_info
                )
        else:
            for id in ids:
                docs_detail = self.vearch.get_doc_by_id(id)
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
