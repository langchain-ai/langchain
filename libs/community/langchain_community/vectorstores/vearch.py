from __future__ import annotations

import json
import uuid
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from vearch.config import Config
from vearch.core.vearch import Vearch
from vearch.schema.field import Field
from vearch.schema.index import HNSWIndex, ScalarIndex
from vearch.schema.space import SpaceSchema
from vearch.utils import DataType, MetricType, VectorInfo

DEFAULT_TOPN = 4

class VearchDb(VectorStore):
    _DEFAULT_TABLE_NAME = "langchain_vearch"
    _DEFAULT_CLUSTER_DB_NAME = "cluster_client_db"

    def __init__(
        self,
        embedding_function: Embeddings,
        path_or_url: Optional[str] = None,
        db_name: str = _DEFAULT_CLUSTER_DB_NAME,
        table_name: str = _DEFAULT_TABLE_NAME,
        **kwargs: Any,
    ) -> None:
        """
        Initialize vearch vector store.
        Please install it with `pip install pyvearch.
        """
    
        if path_or_url is None:
            raise ValueError("Please input router url of vearch")
        if not db_name:
            db_name = self._DEFAULT_CLUSTER_DB_NAME
            db_name += "_"
            db_name += str(uuid.uuid4()).split("-")[-1]
        self.using_db_name = db_name
        self.url = path_or_url
        self.vearch = Vearch(Config(host = path_or_url, token="secret"))
        if not table_name:
            table_name = self._DEFAULT_TABLE_NAME
            table_name += "_"
            table_name += str(uuid.uuid4()).split("-")[-1]
        self.using_table_name = table_name
        self.embedding_func = embedding_function

    @property
    def embeddings(self) -> Optional[Embeddings]:
        return self.embedding_func

    @classmethod
    def from_documents(
        cls: Type[Vearch],
        documents: List[Document],
        embedding: Embeddings,
        path_or_url: Optional[str] = None,
        db_name: str = _DEFAULT_CLUSTER_DB_NAME,
        table_name: str = _DEFAULT_TABLE_NAME,
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
            db_name=db_name,
            table_name=table_name,
            **kwargs,
        )

    @classmethod
    def from_texts(
        cls: Type[Vearch],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        path_or_url: Optional[str] = None,
        db_name: str = _DEFAULT_CLUSTER_DB_NAME,
        table_name: str = _DEFAULT_TABLE_NAME,
        **kwargs: Any,
    ) -> Vearch:
        """Return Vearch VectorStore"""
       
        vearch_db = cls(
            embedding_function=embedding,
            path_or_url=path_or_url,
            db_name=db_name,
            table_name=table_name,
        )
        vearch_db.add_texts(texts=texts, metadatas=metadatas)
        return vearch_db

    def _get_matadata_field(self, metadatas: Optional[List[dict]] = None):
        field_list = []
        if metadatas:
            for key, value in metadatas[0].items():
                if isinstance(value, int):
                    field_list.append({"field": key, "type": "int"})
                    continue
                if isinstance(value, str):
                    field_list.append({"field": key, "type": "str"})
                    continue
                if isinstance(value, float):
                    field_list.append({"field": key, "type": "float"})
                    continue
                else:
                    raise ValueError("Please check data type,support int, str, float")
        return field_list
        

    def _create_space_schema(self, dim) ->SpaceSchema:
        filed_list_add = self.field_list + [{"field": "text", "type": "str"}]
        type_dict = {"int": DataType.INTEGER, "str": DataType.STRING, 
                        "float": DataType.FLOAT}
        fields = [Field("text_embedding", DataType.VECTOR, 
            HNSWIndex("vec_idx", MetricType.Inner_product, 32, 64),dimension=dim)]
        for fi in filed_list_add:
            fields.append(Field(fi["field"], type_dict[fi["type"]], 
                index=ScalarIndex(fi["field"]+"_idx")))
        space_schema = SpaceSchema(self.using_table_name, fields)
        return space_schema


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
        self.field_list = self._get_matadata_field(metadatas)
        dbs= self.vearch.list_databases()
        dbs_list = [item.name["name"] for item in dbs]
        if self.using_db_name not in dbs_list:
            create_db_code = self.vearch.create_database(self.using_db_name)
            if create_db_code.code != 0:
                raise ValueError("create db failed!!!")
        spaces = self.vearch.list_spaces(self.using_db_name)
        space_list = [item.name["space_name"] for item in spaces]
        if self.using_table_name not in space_list:
            create_code = self.vearch.create_space(self.using_db_name, 
                self._create_space_schema(len(embeddings[0])))
            if create_code.code !=0 :
                raise ValueError("create space failed!!!")
        docid = []
        if embeddings is not None and metadatas is not None:
            meta_field_list = [i["field"] for i in self.field_list]
            for text, metadata, embed in zip(texts, metadatas, embeddings):
                profiles: dict[str, Any] = {}
                profiles["text"] = text
                for f in meta_field_list:
                    profiles[f] = metadata[f]
                em_np = np.array(embed)
                profiles["text_embedding"] = (em_np / np.linalg.norm(em_np)).tolist()
                insert_res = self.vearch.upsert(
                    self.using_db_name, self.using_table_name, [profiles]
                )
                if insert_res.code == 0:
                    docid.append(insert_res.document_ids[0]["_id"])
                    continue
                else:
                    retry_insert = self.vearch.upsert(
                        self.using_db_name, self.using_table_name, [profiles]
                    )
                    docid.append(retry_insert.document_ids[0]["_id"])
                    continue
        return docid

    def _get_field_list_from_c(self):

        pass

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
        Returns:
            The k most similar documents to the specified text query.
            0 is dissimilar, 1 is the most similar.
        """

        embed = np.array(embedding)
        _, _, schemas= self.vearch.is_space_exist(
            self.using_db_name, self.using_table_name
        )
        raw_fields = json.loads(schemas)["schema"]["fields"]
        meta_field_list = [item["name"] for item in raw_fields]
        meta_field_list.remove("text_embedding")
        vector = VectorInfo("text_embedding", (embed / np.linalg.norm(embed)).tolist())
        query_result = self.vearch.search(
            self.using_db_name, self.using_table_name, [vector,],
            fields = meta_field_list, limit = k)
        res = query_result.documents[0]
        docs = []
        for item in res:
            content = ""
            meta_data = {}
            for item_key in item:
                if item_key == "text":
                    content = item[item_key]
                    continue
                if item_key in meta_field_list:
                    meta_data[item_key] = item[item_key]
                    meta_field_list.remove(item_key)
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
        _, _, schemas= self.vearch.is_space_exist(
            self.using_db_name, self.using_table_name
        )
        raw_fields = json.loads(schemas)["schema"]["fields"]
        meta_field_list = [item["name"] for item in raw_fields]
        meta_field_list.remove("text_embedding")
        vector = VectorInfo("text_embedding", (embed / np.linalg.norm(embed)).tolist())
        query_result = self.vearch.search(
            self.using_db_name, self.using_table_name, [vector,],
            fields = meta_field_list, limit = k)
        res = query_result.documents[0]
        results: List[Tuple[Document, float]] = []
        for item in res:
            content = ""
            meta_data = {}
            score = item["_score"]
            for item_key in item:
                if item_key == "text":
                    content = item[item_key]
                    continue
                if item_key in meta_field_list:
                    meta_data[item_key] = item[item_key]
                    meta_field_list.remove(item_key)
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

        if ids is None or ids.__len__() == 0:
            return None
        res = self.vearch.delete(self.using_db_name, self.using_table_name, ids)
        if res.code ==0:
            return True
        else:
            return False
       

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
        
        _, _, schemas= self.vearch.is_space_exist(
            self.using_db_name, self.using_table_name
        )
        raw_fields = json.loads(schemas)["schema"]["fields"]
        meta_field_list = [item["name"] for item in raw_fields]
        meta_field_list.remove("text_embedding")

        results: Dict[str, Document] = {}

        if ids is None or ids.__len__() == 0:
            return results
        docs_detail = self.vearch.query(
            self.using_db_name, self.using_table_name, ids
        )
        for record in docs_detail.documents:
            if "code" in record.keys():
                continue
            content = ""
            meta_info = {}
            for field in record:
                if field == "text":
                    content = record[field]
                    continue
                elif field in meta_field_list:
                    meta_info[field] = record[field]
                    meta_field_list.remove(field)
                    continue
            results[record["_id"]] = Document(page_content=content, metadata=meta_info)
        return results
