from __future__ import annotations

import json
import logging
from typing import Any, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

logger = logging.getLogger(__name__)


class Jaguar(VectorStore):
    """`Jaguar API` vector store.

    See http://www.jaguardb.com
    See http://github.com/fserv/jaguar-sdk

    Example:
       .. code-block:: python

           from langchain_community.vectorstores.jaguar import Jaguar

           vectorstore = Jaguar(
               pod = 'vdb',
               store = 'mystore',
               vector_index = 'v',
               vector_type = 'cosine_fraction_float',
               vector_dimension = 1536,
               url='http://192.168.8.88:8080/fwww/',
               embedding=openai_model
           )
    """

    def __init__(
        self,
        pod: str,
        store: str,
        vector_index: str,
        vector_type: str,
        vector_dimension: int,
        url: str,
        embedding: Embeddings,
    ):
        self._pod = pod
        self._store = store
        self._vector_index = vector_index
        self._vector_type = vector_type
        self._vector_dimension = vector_dimension

        self._embedding = embedding
        try:
            from jaguardb_http_client.JaguarHttpClient import JaguarHttpClient
        except ImportError:
            raise ImportError(
                "Could not import jaguardb-http-client python package. "
                "Please install it with `pip install -U jaguardb-http-client`"
            )

        self._jag = JaguarHttpClient(url)
        self._token = ""

    def login(
        self,
        jaguar_api_key: Optional[str] = "",
    ) -> bool:
        """
        login to jaguardb server with a jaguar_api_key or let self._jag find a key
        Args:
            pod (str):  name of a Pod
            store (str):  name of a vector store
            optional jaguar_api_key (str): API key of user to jaguardb server
        Returns:
            True if successful; False if not successful
        """

        if jaguar_api_key == "":
            jaguar_api_key = self._jag.getApiKey()
        self._jaguar_api_key = jaguar_api_key
        self._token = self._jag.login(jaguar_api_key)
        if self._token == "":
            logger.error("E0001 error init(): invalid jaguar_api_key")
            return False
        return True

    def create(
        self,
        metadata_str: str,
        text_size: int,
    ) -> None:
        """
        create the vector store on the backend database
        Args:
            metadata_str (str):  columns and their types
        Returns:
            True if successful; False if not successful
        """
        podstore = self._pod + "." + self._store

        """
        source column is required.
        v:text column is required.
        """
        q = "create store "
        q += podstore
        q += f" ({self._vector_index} vector({self._vector_dimension},"
        q += f" '{self._vector_type}'),"
        q += f" source char(256), v:text char({text_size}),"
        q += metadata_str + ")"
        self.run(q)

    def run(self, query: str, withFile: bool = False) -> dict:
        """
        Run any query statement in jaguardb
        Args:
            query (str): query statement to jaguardb
        Returns:
            None for invalid token, or
            json result string
        """
        if self._token == "":
            logger.error(f"E0005 error run({query})")
            return {}

        resp = self._jag.post(query, self._token, withFile)
        txt = resp.text
        try:
            js = json.loads(txt)
            return js
        except Exception:
            return {}

    @property
    def embeddings(self) -> Optional[Embeddings]:
        return self._embedding

    def add_texts(  # type: ignore[override]
        self,
        texts: List[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """
        Add  texts through the embeddings and add to the vectorstore.
        Args:
          texts: list of text strings to add to the jaguar vector store.
          metadatas: Optional list of metadatas associated with the texts.
            [{"m1": "v11", "m2": "v12", "m3": "v13", "filecol": "path_file1.jpg" },
             {"m1": "v21", "m2": "v22", "m3": "v23", "filecol": "path_file2.jpg" },
             {"m1": "v31", "m2": "v32", "m3": "v33", "filecol": "path_file3.jpg" },
             {"m1": "v41", "m2": "v42", "m3": "v43", "filecol": "path_file4.jpg" }]
          kwargs: vector_index=name_of_vector_index
                  file_column=name_of_file_column

        Returns:
            List of ids from adding the texts into the vectorstore
        """
        vcol = self._vector_index
        filecol = kwargs.get("file_column", "")
        text_tag = kwargs.get("text_tag", "")
        podstorevcol = self._pod + "." + self._store + "." + vcol
        q = "textcol " + podstorevcol
        js = self.run(q)
        if js == "":
            return []
        textcol = js["data"]

        if text_tag != "":
            tag_texts = []
            for t in texts:
                tag_texts.append(text_tag + " " + t)
            texts = tag_texts

        embeddings = self._embedding.embed_documents(list(texts))
        ids = []
        if metadatas is None:
            ### no meta and no files to upload
            i = 0
            for vec in embeddings:
                str_vec = [str(x) for x in vec]
                values_comma = ",".join(str_vec)
                podstore = self._pod + "." + self._store
                q = "insert into " + podstore + " ("
                q += vcol + "," + textcol + ") values ('" + values_comma
                txt = texts[i].replace("'", "\\'")
                q += "','" + txt + "')"
                js = self.run(q, False)
                ids.append(js["zid"])
                i += 1
        else:
            i = 0
            for vec in embeddings:
                str_vec = [str(x) for x in vec]
                nvec, vvec, filepath = self._parseMeta(metadatas[i], filecol)
                if filecol != "":
                    rc = self._jag.postFile(self._token, filepath, 1)
                    if not rc:
                        return []
                names_comma = ",".join(nvec)
                names_comma += "," + vcol
                ## col1,col2,col3,vecl
                values_comma = "'" + "','".join(vvec) + "'"
                ### 'va1','val2','val3'
                values_comma += ",'" + ",".join(str_vec) + "'"
                ### 'v1,v2,v3'
                podstore = self._pod + "." + self._store
                q = "insert into " + podstore + " ("
                q += names_comma + "," + textcol + ") values (" + values_comma
                txt = texts[i].replace("'", "\\'")
                q += ",'" + txt + "')"
                if filecol != "":
                    js = self.run(q, True)
                else:
                    js = self.run(q, False)
                ids.append(js["zid"])
                i += 1

        return ids

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 3,
        fetch_k: int = -1,
        where: Optional[str] = None,
        args: Optional[str] = None,
        metadatas: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """
        Return Jaguar documents most similar to query, along with scores.
        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 3.
            lambda_val: lexical match parameter for hybrid search.
            where: the where clause in select similarity. For example a
                where can be "rating > 3.0 and (state = 'NV' or state = 'CA')"
            args: extra options passed to select similarity
            kwargs:  vector_index=vcol, vector_type=cosine_fraction_float
        Returns:
            List of Documents most similar to the query and score for each.
            List of Tuples of (doc, similarity_score):
                [ (doc, score), (doc, score), ...]
        """
        vcol = self._vector_index
        vtype = self._vector_type
        embeddings = self._embedding.embed_query(query)
        str_embeddings = [str(f) for f in embeddings]
        qv_comma = ",".join(str_embeddings)
        podstore = self._pod + "." + self._store
        q = (
            "select similarity("
            + vcol
            + ",'"
            + qv_comma
            + "','topk="
            + str(k)
            + ",fetch_k="
            + str(fetch_k)
            + ",type="
            + vtype
        )
        q += ",with_score=yes,with_text=yes"
        if args is not None:
            q += "," + args

        if metadatas is not None:
            meta = "&".join(metadatas)
            q += ",metadata=" + meta

        q += "') from " + podstore

        if where is not None:
            q += " where " + where

        jarr = self.run(q)
        if jarr is None:
            return []

        docs_with_score = []
        for js in jarr:
            score = js["score"]
            text = js["text"]
            zid = js["zid"]

            ### give metadatas
            md = {}
            md["zid"] = zid
            if metadatas is not None:
                for m in metadatas:
                    mv = js[m]
                    md[m] = mv

            doc = Document(page_content=text, metadata=md)
            tup = (doc, score)
            docs_with_score.append(tup)

        return docs_with_score

    def similarity_search(
        self,
        query: str,
        k: int = 3,
        where: Optional[str] = None,
        metadatas: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """
        Return Jaguar documents most similar to query, along with scores.
        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 5.
            where: the where clause in select similarity. For example a
                where can be "rating > 3.0 and (state = 'NV' or state = 'CA')"
        Returns:
            List of Documents most similar to the query
        """
        docs_and_scores = self.similarity_search_with_score(
            query, k=k, where=where, metadatas=metadatas, **kwargs
        )
        return [doc for doc, _ in docs_and_scores]

    def is_anomalous(
        self,
        query: str,
        **kwargs: Any,
    ) -> bool:
        """
        Detect if given text is anomalous from the dataset
        Args:
            query: Text to detect if it is anomaly
        Returns:
            True or False
        """
        vcol = self._vector_index
        vtype = self._vector_type
        embeddings = self._embedding.embed_query(query)
        str_embeddings = [str(f) for f in embeddings]
        qv_comma = ",".join(str_embeddings)
        podstore = self._pod + "." + self._store
        q = "select anomalous(" + vcol + ", '" + qv_comma + "', 'type=" + vtype + "')"
        q += " from " + podstore

        js = self.run(q)
        if isinstance(js, list) and len(js) == 0:
            return False
        jd = json.loads(js[0])
        if jd["anomalous"] == "YES":
            return True
        return False

    @classmethod
    def from_texts(  # type: ignore[override]
        cls,
        texts: List[str],
        embedding: Embeddings,
        url: str,
        pod: str,
        store: str,
        vector_index: str,
        vector_type: str,
        vector_dimension: int,
        metadatas: Optional[List[dict]] = None,
        jaguar_api_key: Optional[str] = "",
        **kwargs: Any,
    ) -> Jaguar:
        jagstore = cls(
            pod, store, vector_index, vector_type, vector_dimension, url, embedding
        )
        jagstore.login(jaguar_api_key)
        jagstore.clear()
        jagstore.add_texts(texts, metadatas, **kwargs)
        return jagstore

    def clear(self) -> None:
        """
        Delete all records in jaguardb
        Args: No args
        Returns: None
        """
        podstore = self._pod + "." + self._store
        q = "truncate store " + podstore
        self.run(q)

    def delete(self, zids: List[str], **kwargs: Any) -> None:  # type: ignore[override]
        """
        Delete records in jaguardb by a list of zero-ids
        Args:
            pod (str):  name of a Pod
            ids (List[str]):  a list of zid as string
        Returns:
            Do not return anything
        """
        podstore = self._pod + "." + self._store
        for zid in zids:
            q = "delete from " + podstore + " where zid='" + zid + "'"
            self.run(q)

    def count(self) -> int:
        """
        Count records of a store in jaguardb
        Args: no args
        Returns: (int) number of records in pod store
        """
        podstore = self._pod + "." + self._store
        q = "select count() from " + podstore
        js = self.run(q)
        if isinstance(js, list) and len(js) == 0:
            return 0
        jd = json.loads(js[0])
        return int(jd["data"])

    def drop(self) -> None:
        """
        Drop or remove a store in jaguardb
        Args: no args
        Returns: None
        """
        podstore = self._pod + "." + self._store
        q = "drop store " + podstore
        self.run(q)

    def logout(self) -> None:
        """
        Logout to cleanup resources
        Args: no args
        Returns: None
        """
        self._jag.logout(self._token)

    def prt(self, msg: str) -> None:
        with open("/tmp/debugjaguar.log", "a") as file:
            print(f"msg={msg}", file=file, flush=True)  # noqa: T201

    def _parseMeta(self, nvmap: dict, filecol: str) -> Tuple[List[str], List[str], str]:
        filepath = ""
        if filecol == "":
            nvec = list(nvmap.keys())
            vvec = list(nvmap.values())
        else:
            nvec = []
            vvec = []
            if filecol in nvmap:
                nvec.append(filecol)
                vvec.append(nvmap[filecol])
                filepath = nvmap[filecol]

            for k, v in nvmap.items():
                if k != filecol:
                    nvec.append(k)
                    vvec.append(v)

        vvec_s = [str(e) for e in vvec]
        return nvec, vvec_s, filepath
