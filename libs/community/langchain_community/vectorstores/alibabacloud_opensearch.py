import json
import logging
import numbers
from hashlib import sha1
from typing import Any, Dict, Iterable, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

logger = logging.getLogger()


class AlibabaCloudOpenSearchSettings:
    """Alibaba Cloud Opensearch` client configuration.

    Attribute:
        endpoint (str) : The endpoint of opensearch instance, You can find it
         from the console of Alibaba Cloud OpenSearch.
        instance_id (str) : The identify of opensearch instance, You can find
         it from the console of Alibaba Cloud OpenSearch.
        username (str) : The username specified when purchasing the instance.
        password (str) : The password specified when purchasing the instance，
          After the instance is created, you can modify it on the console.
        tablename (str): The table name specified during instance configuration.
        field_name_mapping (Dict) : Using field name mapping between opensearch
          vector store and opensearch instance configuration table field names:
        {
            'id': 'The id field name map of index document.',
            'document': 'The text field name map of index document.',
            'embedding': 'In the embedding field of the opensearch instance,
              the values must be in float type and separated by separator,
              default is comma.',
            'metadata_field_x': 'Metadata field mapping includes the mapped
             field name and operator in the mapping value, separated by a comma
             between the mapped field name and the operator.',
        }
        protocol (str): Communication Protocol between SDK and Server, default is http.
        namespace (str) : The instance data will be partitioned based on the "namespace"
         field,If the namespace is enabled, you need to specify the namespace field
         name during initialization, Otherwise, the queries cannot be executed
         correctly.
        embedding_field_separator(str): Delimiter specified for writing vector
         field data, default is comma.
        output_fields: Specify the field list returned when invoking OpenSearch,
         by default it is the value list of the field mapping field.
    """

    def __init__(
        self,
        endpoint: str,
        instance_id: str,
        username: str,
        password: str,
        table_name: str,
        field_name_mapping: Dict[str, str],
        protocol: str = "http",
        namespace: str = "",
        embedding_field_separator: str = ",",
        output_fields: Optional[List[str]] = None,
    ) -> None:
        self.endpoint = endpoint
        self.instance_id = instance_id
        self.protocol = protocol
        self.username = username
        self.password = password
        self.namespace = namespace
        self.table_name = table_name
        self.opt_table_name = "_".join([self.instance_id, self.table_name])
        self.field_name_mapping = field_name_mapping
        self.embedding_field_separator = embedding_field_separator
        if output_fields is None:
            self.output_fields = [
                field.split(",")[0] for field in self.field_name_mapping.values()
            ]
        self.inverse_field_name_mapping: Dict[str, str] = {}
        for key, value in self.field_name_mapping.items():
            self.inverse_field_name_mapping[value.split(",")[0]] = key

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
            from alibabacloud_ha3engine_vector import client, models
            from alibabacloud_tea_util import models as util_models
        except ImportError:
            raise ImportError(
                "Could not import alibaba cloud opensearch python package. "
                "Please install it with `pip install alibabacloud-ha3engine-vector`."
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
        self.ha3_engine_client = client.Client(
            models.Config(
                endpoint=config.endpoint,
                instance_id=config.instance_id,
                protocol=config.protocol,
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
        """Insert documents into the instance..
        Args:
            texts: The text segments to be inserted into the vector storage,
             should not be empty.
            metadatas: Metadata information.
        Returns:
            id_list: List of document IDs.
        """

        def _upsert(push_doc_list: List[Dict]) -> List[str]:
            if push_doc_list is None or len(push_doc_list) == 0:
                return []
            try:
                push_request = models.PushDocumentsRequest(
                    self.options_headers, push_doc_list
                )
                push_response = self.ha3_engine_client.push_documents(
                    self.config.opt_table_name, field_name_map["id"], push_request
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

        from alibabacloud_ha3engine_vector import models

        id_list = [sha1(t.encode("utf-8")).hexdigest() for t in texts]
        embeddings = self.embedding.embed_documents(list(texts))
        metadatas = metadatas or [{} for _ in texts]
        field_name_map = self.config.field_name_mapping
        add_doc_list = []
        text_list = list(texts)
        for idx, doc_id in enumerate(id_list):
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
                    self.config.embedding_field_separator.join(
                        str(unit) for unit in embedding
                    ),
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
        """Perform similarity retrieval based on text.
        Args:
            query: Vectorize text for retrieval.，should not be empty.
            k: top n.
            search_filter: Additional filtering conditions.
        Returns:
            document_list: List of documents.
        """
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
        """Perform similarity retrieval based on text with scores.
        Args:
            query: Vectorize text for retrieval.，should not be empty.
            k: top n.
            search_filter: Additional filtering conditions.
        Returns:
            document_list: List of documents.
        """
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
        """Perform retrieval directly using vectors.
        Args:
            embedding: vectors.
            k: top n.
            search_filter: Additional filtering conditions.
        Returns:
            document_list: List of documents.
        """
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
        def generate_filter_query() -> str:
            if search_filter is None:
                return ""
            filter_clause = " AND ".join(
                [
                    create_filter(md_key, md_value)
                    for md_key, md_value in search_filter.items()
                ]
            )
            return filter_clause

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

        def search_data() -> Dict[str, Any]:
            request = QueryRequest(
                table_name=self.config.table_name,
                namespace=self.config.namespace,
                vector=embedding,
                include_vector=True,
                output_fields=self.config.output_fields,
                filter=generate_filter_query(),
                top_k=k,
            )

            query_result = self.ha3_engine_client.query(request)
            return json.loads(query_result.body)

        from alibabacloud_ha3engine_vector.models import QueryRequest

        try:
            json_response = search_data()
            if (
                "errorCode" in json_response
                and "errorMsg" in json_response
                and len(json_response["errorMsg"]) > 0
            ):
                logger.error(
                    f"query {self.config.endpoint} {self.config.instance_id} "
                    f"failed:{json_response['errorMsg']}."
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
        """Assemble documents."""
        items = json_result["result"]
        query_result_list: List[Document] = []
        for item in items:
            if (
                "fields" not in item
                or self.config.field_name_mapping["document"] not in item["fields"]
            ):
                query_result_list.append(Document())  # type: ignore[call-arg]
            else:
                fields = item["fields"]
                query_result_list.append(
                    Document(
                        page_content=fields[self.config.field_name_mapping["document"]],
                        metadata=self.create_inverse_metadata(fields),
                    )
                )
        return query_result_list

    def create_inverse_metadata(self, fields: Dict[str, Any]) -> Dict[str, Any]:
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
            metadata[self.config.inverse_field_name_mapping[key]] = value
        return metadata

    def create_results_with_score(
        self, json_result: Dict[str, Any]
    ) -> List[Tuple[Document, float]]:
        """Parsing the returned results with scores.
        Args:
            json_result: Results from OpenSearch query.
        Returns:
            query_result_list: Results with scores.
        """
        items = json_result["result"]
        query_result_list: List[Tuple[Document, float]] = []
        for item in items:
            fields = item["fields"]
            query_result_list.append(
                (
                    Document(
                        page_content=fields[self.config.field_name_mapping["document"]],
                        metadata=self.create_inverse_metadata(fields),
                    ),
                    float(item["score"]),
                )
            )
        return query_result_list

    def delete_documents_with_texts(self, texts: List[str]) -> bool:
        """Delete documents based on their page content.

        Args:
            texts: List of document page content.
        Returns:
           Whether the deletion was successful or not.
        """
        id_list = [sha1(t.encode("utf-8")).hexdigest() for t in texts]
        return self.delete_documents_with_document_id(id_list)

    def delete_documents_with_document_id(self, id_list: List[str]) -> bool:
        """Delete documents based on their IDs.

        Args:
            id_list: List of document IDs.
        Returns:
            Whether the deletion was successful or not.
        """
        if id_list is None or len(id_list) == 0:
            return True

        from alibabacloud_ha3engine_vector import models

        delete_doc_list = []
        for doc_id in id_list:
            delete_doc_list.append(
                {
                    "fields": {self.config.field_name_mapping["id"]: doc_id},
                    "cmd": "delete",
                }
            )

        delete_request = models.PushDocumentsRequest(
            self.options_headers, delete_doc_list
        )
        try:
            delete_response = self.ha3_engine_client.push_documents(
                self.config.opt_table_name,
                self.config.field_name_mapping["id"],
                delete_request,
            )
            json_response = json.loads(delete_response.body)
            return json_response["status"] == "OK"
        except Exception as e:
            logger.error(
                f"delete doc from :{self.config.endpoint} "
                f"instance_id:{self.config.instance_id} failed.",
                e,
            )
            raise e

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        config: Optional[AlibabaCloudOpenSearchSettings] = None,
        **kwargs: Any,
    ) -> "AlibabaCloudOpenSearch":
        """Create alibaba cloud opensearch vector store instance.

        Args:
            texts: The text segments to be inserted into the vector storage,
             should not be empty.
            embedding: Embedding function, Embedding function.
            config: Alibaba OpenSearch instance configuration.
            metadatas: Metadata information.
        Returns:
            AlibabaCloudOpenSearch: Alibaba cloud opensearch vector store instance.
        """
        if texts is None or len(texts) == 0:
            raise Exception("the inserted text segments, should not be empty.")

        if embedding is None:
            raise Exception("the embeddings should not be empty.")

        if config is None:
            raise Exception("config should not be none.")

        ctx = cls(embedding, config, **kwargs)
        ctx.add_texts(texts=texts, metadatas=metadatas)
        return ctx

    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        embedding: Embeddings,
        config: Optional[AlibabaCloudOpenSearchSettings] = None,
        **kwargs: Any,
    ) -> "AlibabaCloudOpenSearch":
        """Create alibaba cloud opensearch vector store instance.

        Args:
            documents: Documents to be inserted into the vector storage,
             should not be empty.
            embedding: Embedding function, Embedding function.
            config: Alibaba OpenSearch instance configuration.
            ids: Specify the ID for the inserted document. If left empty, the ID will be
             automatically generated based on the text content.
        Returns:
            AlibabaCloudOpenSearch: Alibaba cloud opensearch vector store instance.
        """
        if documents is None or len(documents) == 0:
            raise Exception("the inserted documents, should not be empty.")

        if embedding is None:
            raise Exception("the embeddings should not be empty.")

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
