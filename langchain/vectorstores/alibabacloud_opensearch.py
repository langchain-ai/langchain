from typing import Iterable, Optional, List, Any, Dict, Tuple

from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from langchain.vectorstores import VectorStore
from alibabacloud_ha3engine import models, client
from alibabacloud_tea_util import models as util_models
from hashlib import sha1
import json
import logging
import numbers

logger = logging.getLogger()


class AlibabaCloudOpenSearchSettings:
    """Opensearch Client Configuration
    Attribute:
        endpoint (str) : The endpoint of opensearch instance, You can find it from the console of Alibaba Cloud
                            OpenSearch.
        instance_id (str) : The identify of opensearch instance, You can find it from the console of Alibaba Cloud
                            OpenSearch.
        datasource_name (str): The name of the data source specified when creating it.
        username (str) : The username specified when purchasing the instance.
        password (str) : The password specified when purchasing the instance.
        embedding_index_name (str) :  The name of the vector attribute specified when configuring the instance
                                      attributes.
        field_name_mapping (Dict) : Using field name mapping between opensearch vector store and opensearch instance
                                    configuration table field names:
                                {
                                    'id': 'The id field name map of index document.',
                                    'document': 'The text field name map of index document.',
                                    'embedding': 'In the embedding field of the opensearch instance, the values must be in float16 multivalue type and separated by commas.',
                                    'metadata_field_x': 'Metadata field mapping includes the mapped field name and
                                    operator in the mapping value, separated by a comma between the mapped field name
                                    and the operator.',
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
        "metadata_field_x": "metadata_field_x,operator"
    }

    def __init__(self,
                 endpoint: str,
                 instance_id: str,
                 username: str,
                 password: str,
                 datasource_name: str,
                 embedding_index_name: str,
                 field_name_mapping: Dict[str, str]
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


class AlibabaCloudOpenSearch(VectorStore):

    def __init__(self,
                 embedding: Embeddings,
                 config: AlibabaCloudOpenSearchSettings,
                 **kwargs: Any,
                 ) -> None:
        self.config = config
        self.embedding = embedding

        self.runtime = util_models.RuntimeOptions(
            connect_timeout=5000,
            read_timeout=10000,
            autoretry=False,
            ignore_ssl=False,
            max_idle_conns=50
        )

        self.ha3EngineClient = client.Client(models.Config(
            endpoint=config.endpoint,
            instance_id=config.instance_id,
            protocol="http",
            access_user_name=config.username,
            access_pass_word=config.password
        ))

        self.options_headers = []

    def add_texts(
            self,
            texts: Iterable[str],
            ids: List[str] = None,
            metadatas: Optional[List[dict]] = None,
            **kwargs: Any,
    ) -> List[str]:
        def _upsert(push_doc_list: List[Dict]) -> List[str]:
            if push_doc_list is None or len(push_doc_list) == 0:
                return []
            try:
                push_request = models.PushDocumentsRequestModel(self.options_headers, push_doc_list)
                push_response = self.ha3EngineClient.push_documents(self.config.datasource_name, field_name_map["id"],
                                                                    push_request)
                json_response = json.loads(push_response.body)
                if json_response["status"] == 'OK':
                    return [push_doc["fields"][field_name_map["id"]] for push_doc in push_doc_list]
                return []
            except Exception as e:
                logger.error(
                    f"add doc to alibaba cloud opensearch endpoint:{self.config.endpoint} instance_id:{self.config.instance_id} failed.",
                    e)
                raise e

        ids = ids if ids is not None else [sha1(t.encode("utf-8")).hexdigest() for t in texts]
        embeddings = self.embedding.embed_documents(list(texts))
        metadatas = metadatas or [{} for _ in texts]
        field_name_map = self.config.field_name_mapping
        add_doc_list = []
        for idx, doc_id in enumerate(ids):
            embedding = embeddings[idx] if idx < len(embeddings) else None
            metadata = metadatas[idx] if idx < len(metadatas) else None
            text = texts[idx] if idx < len(texts) else None
            add_doc = dict()
            add_doc_fields = dict()
            add_doc_fields.__setitem__(field_name_map["id"], doc_id)
            add_doc_fields.__setitem__(field_name_map["document"], text)
            add_doc_fields.__setitem__(field_name_map["embedding"], ",".join(str(unit) for unit in embedding))
            for md_key, md_value in metadata.items():
                add_doc_fields.__setitem__(field_name_map[md_key].split(',')[0], md_value)
            add_doc.__setitem__("fields", add_doc_fields)
            add_doc.__setitem__("cmd", "add")
            add_doc_list.append(add_doc)
        return _upsert(add_doc_list)

    def similarity_search(
            self,
            query: str,
            filter: Optional[dict] = None,
            k: int = 4,
            **kwargs: Any,
    ) -> List[Document]:
        return self.inner_search(query=query, filter=filter, k=k, with_score=False, kwargs=kwargs)

    def similarity_search_with_relevance_scores(
            self,
            query: str,
            filter: Optional[dict] = None,
            k: int = 4,
            **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        return self.inner_search(query=query, filter=filter, k=k, with_score=True, kwargs=kwargs)

    def similarity_search_by_vector(
            self,
            query: str,
            filter: Optional[dict] = None,
            k: int = 4,
            **kwargs: Any,
    ) -> List[Document]:
        return self.inner_search(query=query, filter=filter, k=k, with_score=False, kwargs=kwargs)

    def inner_search(
            self,
            query: str,
            filter: Optional[dict] = None,
            k: int = 4,
            with_score: bool = False,
            **kwargs: Any,
    ):
        def generate_query(document: str) -> str:
            embedding = self.embedding.embed_query(document)
            tmp_search_config_str = f"config=start:0,hit:{k},format:json&&cluster=general&&kvpairs=first_formula:proxima_score({self.config.embedding_index_name})&&sort=+RANK"
            tmp_query_str = f"&&query={self.config.embedding_index_name}:" + "'" + ",".join(
                str(x) for x in embedding) + "'"

            if filter is not None:
                filter_clause = "&&filter=" + " AND ".join(
                    [create_filter(md_key, md_value) for md_key, md_value in filter.items()]
                )
                tmp_query_str += filter_clause

            return tmp_search_config_str + tmp_query_str

        def create_filter(md_key, md_value) -> str:
            md_filter_expr = self.config.field_name_mapping[md_key]
            if md_filter_expr is None:
                return None
            exprs = md_filter_expr.split(",")
            if len(exprs) != 2:
                logger.error(
                    f"filter {md_filter_expr} express is not correct, must contain mapping field and operator.")
                return None
            md_filter_key = exprs[0].strip()
            md_filter_operator = exprs[1].strip()
            if isinstance(md_value, numbers.Number):
                return f"{md_filter_key} {md_filter_operator} {md_value}"
            return f"{md_filter_key}{md_filter_operator}\"{md_value}\""

        def search_data(single_query_str: str):
            search_query = models.SearchQuery(query=single_query_str)
            search_request = models.SearchRequestModel(self.options_headers, search_query)
            return self.ha3EngineClient.search(search_request)

        def create_results(json_result) -> List[Document]:
            items = json_result['result']['items']
            query_result_list: List[Document] = []
            for item in items:
                fields = item["fields"]
                query_result_list.append(Document(page_content=fields[self.config.field_name_mapping["document"]],
                                                  metadata=create_metadata(fields)))
            return query_result_list

        def create_results_with_score(json_result) -> List[Tuple[Document, float]]:
            items = json_result['result']['items']
            query_result_list: List[Tuple[Document, float]] = []
            for item in items:
                fields = item["fields"]
                query_result_list.append((Document(page_content=fields[self.config.field_name_mapping["document"]],
                                                   metadata=create_metadata(fields)), float(item["sortExprValues"][0])))
            return query_result_list

        def create_metadata(fields: dict) -> dict:
            metadata: dict = {}
            for key, value in fields.items():
                if key == "id" or key == "document" or key == "embedding":
                    continue
                metadata[key] = value
            return metadata

        def single_query(query: str, with_score: bool) -> List[Document]:
            config = self.config
            try:
                query_str = generate_query(query)
                search_response = search_data(query_str)
                json_response = json.loads(search_response.body)
                if len(json_response["errors"]) != 0:
                    logger.error(
                        f"query {config.endpoint} {config.instance_id} errors:{json_response['errors']} failed.")
                else:
                    return create_results_with_score(json_response) if with_score else create_results(json_response)
            except Exception as e:
                logger.error(
                    f"query alibaba cloud opensearch endpoint:{config.endpoint} instance_id:{config.instance_id} failed.",
                    e)
            return []

        return single_query(query, with_score)

    @classmethod
    def from_texts(
            cls,
            texts: List[str],
            embedding: Embeddings,
            metadatas: Optional[List[Dict[Any, Any]]],
            config: Optional[AlibabaCloudOpenSearchSettings],
            text_ids: Optional[Iterable[str]] = None,
            **kwargs: Any
    ):
        ctx = cls(embedding, config, **kwargs)
        ctx.add_texts(texts, ids=text_ids, metadatas=metadatas)
        return ctx

    @classmethod
    def from_documents(
            cls,
            documents: List[Document],
            embedding: Embeddings,
            ids: Optional[List[str]] = None,
            **kwargs: Any,
    ):
        texts = [d.page_content for d in documents]
        metadatas = [d.metadata for d in documents]

        return cls.from_texts(
            texts=texts,
            embedding=embedding,
            metadatas=metadatas,
            ids=ids,
            **kwargs,
        )
