import json
import logging
import numbers
from hashlib import sha1
from typing import Any, Dict, Iterable, List, Optional, Tuple

from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from langchain.vectorstores.base import VectorStore

logger = logging.getLogger()


class AlibabaCloudOpenSearchSettings:
    """`Alibaba Cloud Opensearch` client configuration.

    Attribute:
        endpoint (str) : The endpoint of opensearch instance, You can find it
          from the console of Alibaba Cloud OpenSearch.
        instance_id (str) : The identify of opensearch instance, You can find
          it from the console of Alibaba Cloud OpenSearch.
        datasource_name (str): The name of the data source specified when creating it.
        username (str) : The username specified when purchasing the instance.
        password (str) : The password specified when purchasing the instance.
        embedding_index_name (str) :  The name of the vector attribute specified
          when configuring the instance attributes.
        field_name_mapping (Dict) : Using field name mapping between opensearch
          vector store and opensearch instance configuration table field names:
        {
            'id': 'The id field name map of index document.',
            'document': 'The text field name map of index document.',
            'embedding': 'In the embedding field of the opensearch instance,
            the values must be in float16 multivalue type and separated by commas.',
            'metadata_field_x': 'Metadata field mapping includes the mapped
            field name and operator in the mapping value, separated by a comma
            between the mapped field name and the operator.',
        }
    """

    endpoint: str
    instance_id: str
    username: str
    password: str
    datasource_name: str
    embedding_index_name: str
    field_name_mapping: Dict[str, str] = {
        "id": "id",
        "document": "document",
        "embedding": "embedding",
        "metadata_field_x": "metadata_field_x,operator",
    }

    def __init__(
        self,
        endpoint: str,
        instance_id: str,
        username: str,
        password: str,
        datasource_name: str,
        embedding_index_name: str,
        field_name_mapping: Dict[str, str],
    ) -> None:
        self.endpoint = endpoint
        self.instance_id = instance_id
        self.username = username
        self.password = password
        self.datasource_name = datasource_name
        self.embedding_index_name = embedding_index_name
        self.field_name_mapping = field_name_mapping

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)


def create_metadata(fields: Dict[str, Any]) -> Dict[str, Any]:
    """Create metadata from fields.

    Args:
        fields: The fields of the document. The fields must be a dict.

    Returns:
        metadata: The metadata of the document. The metadata must be a dict.
    """
    metadata: Dict[str, Any] = {}
    for key, value in fields.items():
        if key == "id" or key == "document" or key == "embedding":
            continue
        metadata[key] = value
    return metadata


class AlibabaCloudOpenSearch(VectorStore):
    """`Alibaba Cloud OpenSearch` vector store."""

    def __init__(
        self,
        embedding: Embeddings,
        config: AlibabaCloudOpenSearchSettings,
        **kwargs: Any,
    ) -> None:
        try:
            from alibabacloud_ha3engine import client, models
            from alibabacloud_tea_util import models as util_models
        except ImportError:
            raise ImportError(
                "Could not import alibaba cloud opensearch python package. "
                "Please install it with `pip install alibabacloud-ha3engine`."
            )

        self.config = config
        self.embedding = embedding

        self.runtime = util_models.RuntimeOptions(
            connect_timeout=5000,
            read_timeout=10000,
            autoretry=False,
            ignore_ssl=False,
            max_idle_conns=50,
        )
        self.ha3EngineClient = client.Client(
            models.Config(
                endpoint=config.endpoint,
                instance_id=config.instance_id,
                protocol="http",
                access_user_name=config.username,
                access_pass_word=config.password,
            )
        )

        self.options_headers: Dict[str, str] = {}

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        def _upsert(push_doc_list: List[Dict]) -> List[str]:
            if push_doc_list is None or len(push_doc_list) == 0:
                return []
            try:
                push_request = models.PushDocumentsRequestModel(
                    self.options_headers, push_doc_list
                )
                push_response = self.ha3EngineClient.push_documents(
                    self.config.datasource_name, field_name_map["id"], push_request
                )
                json_response = json.loads(push_response.body)
                if json_response["status"] == "OK":
                    return [
                        push_doc["fields"][field_name_map["id"]]
                        for push_doc in push_doc_list
                    ]
                return []
            except Exception as e:
                logger.error(
                    f"add doc to endpoint:{self.config.endpoint} "
                    f"instance_id:{self.config.instance_id} failed.",
                    e,
                )
                raise e

        from alibabacloud_ha3engine import models

        ids = [sha1(t.encode("utf-8")).hexdigest() for t in texts]
        embeddings = self.embedding.embed_documents(list(texts))
        metadatas = metadatas or [{} for _ in texts]
        field_name_map = self.config.field_name_mapping
        add_doc_list = []
        text_list = list(texts)
        for idx, doc_id in enumerate(ids):
            embedding = embeddings[idx] if idx < len(embeddings) else None
            metadata = metadatas[idx] if idx < len(metadatas) else None
            text = text_list[idx] if idx < len(text_list) else None
            add_doc: Dict[str, Any] = dict()
            add_doc_fields: Dict[str, Any] = dict()
            add_doc_fields.__setitem__(field_name_map["id"], doc_id)
            add_doc_fields.__setitem__(field_name_map["document"], text)
            if embedding is not None:
                add_doc_fields.__setitem__(
                    field_name_map["embedding"],
                    ",".join(str(unit) for unit in embedding),
                )
            if metadata is not None:
                for md_key, md_value in metadata.items():
                    add_doc_fields.__setitem__(
                        field_name_map[md_key].split(",")[0], md_value
                    )
            add_doc.__setitem__("fields", add_doc_fields)
            add_doc.__setitem__("cmd", "add")
            add_doc_list.append(add_doc)
        return _upsert(add_doc_list)

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        search_filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        embedding = self.embedding.embed_query(query)
        return self.create_results(
            self.inner_embedding_query(
                embedding=embedding, search_filter=search_filter, k=k
            )
        )

    def similarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 4,
        search_filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        embedding: List[float] = self.embedding.embed_query(query)
        return self.create_results_with_score(
            self.inner_embedding_query(
                embedding=embedding, search_filter=search_filter, k=k
            )
        )

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        search_filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Document]:
        return self.create_results(
            self.inner_embedding_query(
                embedding=embedding, search_filter=search_filter, k=k
            )
        )

    def inner_embedding_query(
        self,
        embedding: List[float],
        search_filter: Optional[Dict[str, Any]] = None,
        k: int = 4,
    ) -> Dict[str, Any]:
        def generate_embedding_query() -> str:
            tmp_search_config_str = (
                f"config=start:0,hit:{k},format:json&&cluster=general&&kvpairs="
                f"first_formula:proxima_score({self.config.embedding_index_name})&&sort=+RANK"
            )
            tmp_query_str = (
                f"&&query={self.config.embedding_index_name}:"
                + "'"
                + ",".join(str(x) for x in embedding)
                + "'"
            )
            if search_filter is not None:
                filter_clause = "&&filter=" + " AND ".join(
                    [
                        create_filter(md_key, md_value)
                        for md_key, md_value in search_filter.items()
                    ]
                )
                tmp_query_str += filter_clause

            return tmp_search_config_str + tmp_query_str

        def create_filter(md_key: str, md_value: Any) -> str:
            md_filter_expr = self.config.field_name_mapping[md_key]
            if md_filter_expr is None:
                return ""
            expr = md_filter_expr.split(",")
            if len(expr) != 2:
                logger.error(
                    f"filter {md_filter_expr} express is not correct, "
                    f"must contain mapping field and operator."
                )
                return ""
            md_filter_key = expr[0].strip()
            md_filter_operator = expr[1].strip()
            if isinstance(md_value, numbers.Number):
                return f"{md_filter_key} {md_filter_operator} {md_value}"
            return f'{md_filter_key}{md_filter_operator}"{md_value}"'

        def search_data(single_query_str: str) -> Dict[str, Any]:
            search_query = models.SearchQuery(query=single_query_str)
            search_request = models.SearchRequestModel(
                self.options_headers, search_query
            )
            return json.loads(self.ha3EngineClient.search(search_request).body)

        from alibabacloud_ha3engine import models

        try:
            query_str = generate_embedding_query()
            json_response = search_data(query_str)
            if len(json_response["errors"]) != 0:
                logger.error(
                    f"query {self.config.endpoint} {self.config.instance_id} "
                    f"errors:{json_response['errors']} failed."
                )
            else:
                return json_response
        except Exception as e:
            logger.error(
                f"query instance endpoint:{self.config.endpoint} "
                f"instance_id:{self.config.instance_id} failed.",
                e,
            )
        return {}

    def create_results(self, json_result: Dict[str, Any]) -> List[Document]:
        items = json_result["result"]["items"]
        query_result_list: List[Document] = []
        for item in items:
            fields = item["fields"]
            query_result_list.append(
                Document(
                    page_content=fields[self.config.field_name_mapping["document"]],
                    metadata=create_metadata(fields),
                )
            )
        return query_result_list

    def create_results_with_score(
        self, json_result: Dict[str, Any]
    ) -> List[Tuple[Document, float]]:
        items = json_result["result"]["items"]
        query_result_list: List[Tuple[Document, float]] = []
        for item in items:
            fields = item["fields"]
            query_result_list.append(
                (
                    Document(
                        page_content=fields[self.config.field_name_mapping["document"]],
                        metadata=create_metadata(fields),
                    ),
                    float(item["sortExprValues"][0]),
                )
            )
        return query_result_list

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        config: Optional[AlibabaCloudOpenSearchSettings] = None,
        **kwargs: Any,
    ) -> "AlibabaCloudOpenSearch":
        if config is None:
            raise Exception("config can't be none")

        ctx = cls(embedding, config, **kwargs)
        ctx.add_texts(texts=texts, metadatas=metadatas)
        return ctx

    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        embedding: Embeddings,
        ids: Optional[List[str]] = None,
        config: Optional[AlibabaCloudOpenSearchSettings] = None,
        **kwargs: Any,
    ) -> "AlibabaCloudOpenSearch":
        if config is None:
            raise Exception("config can't be none")

        texts = [d.page_content for d in documents]
        metadatas = [d.metadata for d in documents]
        return cls.from_texts(
            texts=texts,
            embedding=embedding,
            metadatas=metadatas,
            config=config,
            **kwargs,
        )
