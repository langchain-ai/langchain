"""Module providing Infinispan as a VectorStore"""

from __future__ import annotations

import json
import logging
import uuid
import warnings
from typing import Any, Iterable, List, Optional, Tuple, Type, Union, cast

from httpx import Response
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

logger = logging.getLogger(__name__)


class InfinispanVS(VectorStore):
    """`Infinispan` VectorStore interface.

        This class exposes the method to present Infinispan as a
        VectorStore. It relies on the Infinispan class (below) which takes care
        of the REST interface with the server.

    Example:
        ... code-block:: python
            from langchain_community.vectorstores import InfinispanVS
            from mymodels import RGBEmbeddings
            ...
            vectorDb = InfinispanVS.from_documents(docs,
                            embedding=RGBEmbeddings(),
                            output_fields=["texture", "color"],
                            lambda_key=lambda text,meta: str(meta["_key"]),
                            lambda_content=lambda item: item["color"])

        or an empty InfinispanVS instance can be created if preliminary setup
        is required before populating the store

        ... code-block:: python
            from langchain_community.vectorstores import InfinispanVS
            from mymodels import RGBEmbeddings
            ...
            ispnVS = InfinispanVS()
            # configure Infinispan here
            # i.e. create cache and schema

            # then populate the store
            vectorDb = InfinispanVS.from_documents(docs,
                            embedding=RGBEmbeddings(),
                            output_fields: ["texture", "color"],
                            lambda_key: lambda text,meta: str(meta["_key"]),
                            lambda_content: lambda item: item["color"])
    """

    def __init__(
        self,
        embedding: Optional[Embeddings] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        """
        Parameters
        ----------
        cache_name: str
            Embeddings cache name. Default "vector"
        entity_name: str
            Protobuf entity name for the embeddings. Default "vector"
        text_field: str
            Protobuf field name for text. Default "text"
        vector_field: str
            Protobuf field name for vector. Default "vector"
        lambda_content: lambda
            Lambda returning the content part of an item. Default returns text_field
        lambda_metadata: lambda
            Lambda returning the metadata part of an item. Default returns items
            fields excepts text_field, vector_field, _type
        output_fields: List[str]
            List of fields to be returned from item, if None return all fields.
            Default None
        kwargs: Any
            Rest of arguments passed to Infinispan. See docs"""
        self.ispn = Infinispan(**kwargs)
        self._configuration = kwargs
        self._cache_name = str(self._configuration.get("cache_name", "vector"))
        self._entity_name = str(self._configuration.get("entity_name", "vector"))
        self._embedding = embedding
        self._textfield = self._configuration.get("textfield", "")
        if self._textfield == "":
            self._textfield = self._configuration.get("text_field", "text")
        else:
            warnings.warn(
                "`textfield` is deprecated. Please use `text_field` " "param.",
                DeprecationWarning,
            )
        self._vectorfield = self._configuration.get("vectorfield", "")
        if self._vectorfield == "":
            self._vectorfield = self._configuration.get("vector_field", "vector")
        else:
            warnings.warn(
                "`vectorfield` is deprecated. Please use `vector_field` " "param.",
                DeprecationWarning,
            )
        self._to_content = self._configuration.get(
            "lambda_content", lambda item: self._default_content(item)
        )
        self._to_metadata = self._configuration.get(
            "lambda_metadata", lambda item: self._default_metadata(item)
        )
        self._output_fields = self._configuration.get("output_fields")
        self._ids = ids

    def _default_metadata(self, item: dict) -> dict:
        meta = dict(item)
        meta.pop(self._vectorfield, None)
        meta.pop(self._textfield, None)
        meta.pop("_type", None)
        return meta

    def _default_content(self, item: dict[str, Any]) -> Any:
        return item.get(self._textfield)

    def schema_builder(self, templ: dict, dimension: int) -> str:
        metadata_proto_tpl = """
/**
* @Indexed
*/
message %s {
/**
* @Vector(dimension=%d)
*/
repeated float %s = 1;
"""
        metadata_proto = metadata_proto_tpl % (
            self._entity_name,
            dimension,
            self._vectorfield,
        )
        idx = 2
        for f, v in templ.items():
            if isinstance(v, str):
                metadata_proto += "optional string " + f + " = " + str(idx) + ";\n"
            elif isinstance(v, int):
                metadata_proto += "optional int64 " + f + " = " + str(idx) + ";\n"
            elif isinstance(v, float):
                metadata_proto += "optional double " + f + " = " + str(idx) + ";\n"
            elif isinstance(v, bytes):
                metadata_proto += "optional bytes " + f + " = " + str(idx) + ";\n"
            elif isinstance(v, bool):
                metadata_proto += "optional bool " + f + " = " + str(idx) + ";\n"
            else:
                raise Exception(
                    "Unable to build proto schema for metadata. "
                    "Unhandled type for field: " + f
                )
            idx += 1
        metadata_proto += "}\n"
        return metadata_proto

    def schema_create(self, proto: str) -> Response:
        """Deploy the schema for the vector db
        Args:
            proto(str): protobuf schema
        Returns:
            An http Response containing the result of the operation
        """
        return self.ispn.schema_post(self._entity_name + ".proto", proto)

    def schema_delete(self) -> Response:
        """Delete the schema for the vector db
        Returns:
            An http Response containing the result of the operation
        """
        return self.ispn.schema_delete(self._entity_name + ".proto")

    def cache_create(self, config: str = "") -> Response:
        """Create the cache for the vector db
        Args:
            config(str): configuration of the cache.
        Returns:
            An http Response containing the result of the operation
        """
        if config == "":
            config = (
                '''
            {
  "distributed-cache": {
    "owners": "2",
    "mode": "SYNC",
    "statistics": true,
    "encoding": {
      "media-type": "application/x-protostream"
    },
    "indexing": {
      "enabled": true,
      "storage": "filesystem",
      "startup-mode": "AUTO",
      "indexing-mode": "AUTO",
      "indexed-entities": [
        "'''
                + self._entity_name
                + """"
      ]
    }
  }
}
"""
            )
        return self.ispn.cache_post(self._cache_name, config)

    def cache_delete(self) -> Response:
        """Delete the cache for the vector db
        Returns:
            An http Response containing the result of the operation
        """
        return self.ispn.cache_delete(self._cache_name)

    def cache_clear(self) -> Response:
        """Clear the cache for the vector db
        Returns:
            An http Response containing the result of the operation
        """
        return self.ispn.cache_clear(self._cache_name)

    def cache_exists(self) -> bool:
        """Checks if the cache exists
        Returns:
            true if exists
        """
        return self.ispn.cache_exists(self._cache_name)

    def cache_index_clear(self) -> Response:
        """Clear the index for the vector db
        Returns:
            An http Response containing the result of the operation
        """
        return self.ispn.index_clear(self._cache_name)

    def cache_index_reindex(self) -> Response:
        """Rebuild the for the vector db
        Returns:
            An http Response containing the result of the operation
        """
        return self.ispn.index_reindex(self._cache_name)

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        last_vector: Optional[List[float]] = None,
        **kwargs: Any,
    ) -> List[str]:
        result = []
        texts_l = list(texts)
        if last_vector:
            texts_l.pop()
        embeds = self._embedding.embed_documents(texts_l)  # type: ignore
        if last_vector:
            embeds.append(last_vector)
        if not metadatas:
            metadatas = [{} for _ in texts]
        ids = self._ids or [str(uuid.uuid4()) for _ in texts]
        data_input = list(zip(metadatas, embeds, ids))
        for metadata, embed, key in data_input:
            data = {"_type": self._entity_name, self._vectorfield: embed}
            data.update(metadata)
            data_str = json.dumps(data)
            self.ispn.put(key, data_str, self._cache_name)
            result.append(key)
        return result

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Return docs most similar to query."""
        documents = self.similarity_search_with_score(query=query, k=k)
        return [doc for doc, _ in documents]

    def similarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """Perform a search on a query string and return results with score.

        Args:
            query (str): The text being searched.
            k (int, optional): The amount of results to return. Defaults to 4.

        Returns:
            List[Tuple[Document, float]]
        """
        embed = self._embedding.embed_query(query)  # type: ignore
        documents = self.similarity_search_with_score_by_vector(embedding=embed, k=k)
        return documents

    def similarity_search_by_vector(
        self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Document]:
        res = self.similarity_search_with_score_by_vector(embedding, k)
        return [doc for doc, _ in res]

    def similarity_search_with_score_by_vector(
        self, embedding: List[float], k: int = 4
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.

        Returns:
            List of pair (Documents, score) most similar to the query vector.
        """
        if self._output_fields is None:
            query_str = (
                "select v, score(v) from "
                + self._entity_name
                + " v where v."
                + self._vectorfield
                + " <-> "
                + json.dumps(embedding)
                + "~"
                + str(k)
            )
        else:
            query_proj = "select "
            for field in self._output_fields[:-1]:
                query_proj = query_proj + "v." + field + ","
            query_proj = query_proj + "v." + self._output_fields[-1]
            query_str = (
                query_proj
                + ", score(v) from "
                + self._entity_name
                + " v where v."
                + self._vectorfield
                + " <-> "
                + json.dumps(embedding)
                + "~"
                + str(k)
            )
        query_res = self.ispn.req_query(query_str, self._cache_name)
        result = json.loads(query_res.text)
        return self._query_result_to_docs(result)

    def _query_result_to_docs(
        self, result: dict[str, Any]
    ) -> List[Tuple[Document, float]]:
        documents = []
        for row in result["hits"]:
            hit = row["hit"] or {}
            if self._output_fields is None:
                entity = hit["*"]
            else:
                entity = {key: hit.get(key) for key in self._output_fields}
            doc = Document(
                page_content=self._to_content(entity),
                metadata=self._to_metadata(entity),
            )
            documents.append((doc, hit["score()"]))
        return documents

    def configure(self, metadata: dict, dimension: int) -> None:
        schema = self.schema_builder(metadata, dimension)
        output = self.schema_create(schema)
        assert (
            output.status_code == self.ispn.Codes.OK
        ), "Unable to create schema. Already exists? "
        "Consider using clear_old=True"
        assert json.loads(output.text)["error"] is None
        if not self.cache_exists():
            output = self.cache_create()
            assert (
                output.status_code == self.ispn.Codes.OK
            ), "Unable to create cache. Already exists? "
            "Consider using clear_old=True"
            # Ensure index is clean
            self.cache_index_clear()

    def config_clear(self) -> None:
        self.schema_delete()
        self.cache_delete()

    @classmethod
    def from_texts(
        cls: Type[InfinispanVS],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        clear_old: Optional[bool] = True,
        auto_config: Optional[bool] = True,
        **kwargs: Any,
    ) -> InfinispanVS:
        """Return VectorStore initialized from texts and embeddings.

        In addition to parameters described by the super method, this
        implementation provides other configuration params if different
        configuration from default is needed.

        Parameters
        ----------
        ids : List[str]
            Additional list of keys associated to the embedding. If not
            provided UUIDs will be generated
        clear_old : bool
            Whether old data must be deleted. Default True
        auto_config: bool
            Whether to do a complete server setup (caches,
            protobuf definition...). Default True
        kwargs: Any
            Rest of arguments passed to InfinispanVS. See docs"""
        infinispanvs = cls(embedding=embedding, ids=ids, **kwargs)
        if auto_config and len(metadatas or []) > 0:
            if clear_old:
                infinispanvs.config_clear()
            vec = embedding.embed_query(texts[len(texts) - 1])
            metadatas = cast(List[dict], metadatas)
            infinispanvs.configure(metadatas[0], len(vec))
        else:
            if clear_old:
                infinispanvs.cache_clear()
            vec = embedding.embed_query(texts[len(texts) - 1])
        if texts:
            infinispanvs.add_texts(texts, metadatas, vector=vec)
        return infinispanvs


REST_TIMEOUT = 10


class Infinispan:
    """Helper class for `Infinispan` REST interface.

    This class exposes the Infinispan operations needed to
    create and set up a vector db.

    You need a running Infinispan (15+) server without authentication.
    You can easily start one, see:
    https://github.com/rigazilla/infinispan-vector#run-infinispan
    """

    def __init__(
        self,
        schema: str = "http",
        user: str = "",
        password: str = "",
        hosts: List[str] = ["127.0.0.1:11222"],
        cache_url: str = "/rest/v2/caches",
        schema_url: str = "/rest/v2/schemas",
        use_post_for_query: bool = True,
        http2: bool = True,
        verify: bool = True,
        **kwargs: Any,
    ):
        """
        Parameters
        ----------
        schema: str
            Schema for HTTP request: "http" or "https". Default "http"
        user, password: str
            User and password if auth is required. Default None
        hosts: List[str]
            List of server addresses. Default ["127.0.0.1:11222"]
        cache_url: str
            URL endpoint for cache API. Default "/rest/v2/caches"
        schema_url: str
            URL endpoint for schema API. Default "/rest/v2/schemas"
        use_post_for_query: bool
            Whether POST method should be used for query. Default True
        http2: bool
            Whether HTTP/2 protocol should be used. `pip install "httpx[http2]"` is
            needed for HTTP/2. Default True
        verify:  bool
            Whether TLS certificate must be verified. Default True
        """

        try:
            import httpx
        except ImportError:
            raise ImportError(
                "Could not import httpx python package. "
                "Please install it with `pip install httpx`"
                'or `pip install "httpx[http2]"` if you need HTTP/2.'
            )

        self.Codes = httpx.codes

        self._configuration = kwargs
        self._schema = schema
        self._user = user
        self._password = password
        self._host = hosts[0]
        self._default_node = self._schema + "://" + self._host
        self._cache_url = cache_url
        self._schema_url = schema_url
        self._use_post_for_query = use_post_for_query
        self._http2 = http2
        if self._user and self._password:
            if self._schema == "http":
                auth: Union[Tuple[str, str], httpx.DigestAuth] = httpx.DigestAuth(
                    username=self._user, password=self._password
                )
            else:
                auth = (self._user, self._password)
            self._h2c = httpx.Client(
                http2=self._http2,
                http1=not self._http2,
                auth=auth,
                verify=verify,
            )
        else:
            self._h2c = httpx.Client(
                http2=self._http2,
                http1=not self._http2,
                verify=verify,
            )

    def req_query(self, query: str, cache_name: str, local: bool = False) -> Response:
        """Request a query
        Args:
            query(str): query requested
            cache_name(str): name of the target cache
            local(boolean): whether the query is local to clustered
        Returns:
            An http Response containing the result set or errors
        """
        if self._use_post_for_query:
            return self._query_post(query, cache_name, local)
        return self._query_get(query, cache_name, local)

    def _query_post(
        self, query_str: str, cache_name: str, local: bool = False
    ) -> Response:
        api_url = (
            self._default_node
            + self._cache_url
            + "/"
            + cache_name
            + "?action=search&local="
            + str(local)
        )
        data = {"query": query_str}
        data_json = json.dumps(data)
        response = self._h2c.post(
            api_url,
            content=data_json,
            headers={"Content-Type": "application/json"},
            timeout=REST_TIMEOUT,
        )
        return response

    def _query_get(
        self, query_str: str, cache_name: str, local: bool = False
    ) -> Response:
        api_url = (
            self._default_node
            + self._cache_url
            + "/"
            + cache_name
            + "?action=search&query="
            + query_str
            + "&local="
            + str(local)
        )
        response = self._h2c.get(api_url, timeout=REST_TIMEOUT)
        return response

    def post(self, key: str, data: str, cache_name: str) -> Response:
        """Post an entry
        Args:
            key(str): key of the entry
            data(str): content of the entry in json format
            cache_name(str): target cache
        Returns:
            An http Response containing the result of the operation
        """
        api_url = self._default_node + self._cache_url + "/" + cache_name + "/" + key
        response = self._h2c.post(
            api_url,
            content=data,
            headers={"Content-Type": "application/json"},
            timeout=REST_TIMEOUT,
        )
        return response

    def put(self, key: str, data: str, cache_name: str) -> Response:
        """Put an entry
        Args:
            key(str): key of the entry
            data(str): content of the entry in json format
            cache_name(str): target cache
        Returns:
            An http Response containing the result of the operation
        """
        api_url = self._default_node + self._cache_url + "/" + cache_name + "/" + key
        response = self._h2c.put(
            api_url,
            content=data,
            headers={"Content-Type": "application/json"},
            timeout=REST_TIMEOUT,
        )
        return response

    def get(self, key: str, cache_name: str) -> Response:
        """Get an entry
        Args:
            key(str): key of the entry
            cache_name(str): target cache
        Returns:
            An http Response containing the entry or errors
        """
        api_url = self._default_node + self._cache_url + "/" + cache_name + "/" + key
        response = self._h2c.get(
            api_url, headers={"Content-Type": "application/json"}, timeout=REST_TIMEOUT
        )
        return response

    def schema_post(self, name: str, proto: str) -> Response:
        """Deploy a schema
        Args:
            name(str): name of the schema. Will be used as a key
            proto(str): protobuf schema
        Returns:
            An http Response containing the result of the operation
        """
        api_url = self._default_node + self._schema_url + "/" + name
        response = self._h2c.post(api_url, content=proto, timeout=REST_TIMEOUT)
        return response

    def cache_post(self, name: str, config: str) -> Response:
        """Create a cache
        Args:
            name(str): name of the cache.
            config(str): configuration of the cache.
        Returns:
            An http Response containing the result of the operation
        """
        api_url = self._default_node + self._cache_url + "/" + name
        response = self._h2c.post(
            api_url,
            content=config,
            headers={"Content-Type": "application/json"},
            timeout=REST_TIMEOUT,
        )
        return response

    def schema_delete(self, name: str) -> Response:
        """Delete a schema
        Args:
            name(str): name of the schema.
        Returns:
            An http Response containing the result of the operation
        """
        api_url = self._default_node + self._schema_url + "/" + name
        response = self._h2c.delete(api_url, timeout=REST_TIMEOUT)
        return response

    def cache_delete(self, name: str) -> Response:
        """Delete a cache
        Args:
            name(str): name of the cache.
        Returns:
            An http Response containing the result of the operation
        """
        api_url = self._default_node + self._cache_url + "/" + name
        response = self._h2c.delete(api_url, timeout=REST_TIMEOUT)
        return response

    def cache_clear(self, cache_name: str) -> Response:
        """Clear a cache
        Args:
            cache_name(str): name of the cache.
        Returns:
            An http Response containing the result of the operation
        """
        api_url = (
            self._default_node + self._cache_url + "/" + cache_name + "?action=clear"
        )
        response = self._h2c.post(api_url, timeout=REST_TIMEOUT)
        return response

    def cache_exists(self, cache_name: str) -> bool:
        """Check if a cache exists
        Args:
            cache_name(str): name of the cache.
        Returns:
            True if cache exists
        """
        api_url = (
            self._default_node + self._cache_url + "/" + cache_name + "?action=clear"
        )
        return self.resource_exists(api_url)

    def resource_exists(self, api_url: str) -> bool:
        """Check if a resource exists
        Args:
            api_url(str): url of the resource.
        Returns:
            true if resource exists
        """
        response = self._h2c.head(api_url, timeout=REST_TIMEOUT)
        return response.status_code == self.Codes.OK

    def index_clear(self, cache_name: str) -> Response:
        """Clear an index on a cache
        Args:
            cache_name(str): name of the cache.
        Returns:
            An http Response containing the result of the operation
        """
        api_url = (
            self._default_node
            + self._cache_url
            + "/"
            + cache_name
            + "/search/indexes?action=clear"
        )
        return self._h2c.post(api_url, timeout=REST_TIMEOUT)

    def index_reindex(self, cache_name: str) -> Response:
        """Rebuild index on a cache
        Args:
            cache_name(str): name of the cache.
        Returns:
            An http Response containing the result of the operation
        """
        api_url = (
            self._default_node
            + self._cache_url
            + "/"
            + cache_name
            + "/search/indexes?action=reindex"
        )
        return self._h2c.post(api_url, timeout=REST_TIMEOUT)
