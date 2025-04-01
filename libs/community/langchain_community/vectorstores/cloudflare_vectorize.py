from __future__ import annotations

import asyncio
import time
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union
)
import uuid
import requests
import json
import inspect
from typing_extensions import TypedDict
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

WAIT_SECONDS = 5
MAX_INSERT_SIZE = 1000
DEFAULT_TOP_K = 20
DEFAULT_TOP_K_WITH_MD_VALUES = 20
DEFAULT_DIMENSIONS = 1024
DEFAULT_METRIC = "cosine"

# Type variable for class methods that return CloudflareVectorize
VST = TypeVar("VST", bound="CloudflareVectorize")


# MARK: - RequestsKwargs
class RequestsKwargs(TypedDict, total=False):
    """TypedDict for requests kwargs."""
    timeout: Union[float, Tuple[float, float], None]
    verify: Union[bool, str]
    cert: Union[str, Tuple[str, str], None]
    proxies: Optional[Dict[str, str]]
    allow_redirects: bool
    stream: bool
    params: Optional[Dict[str, Any]]


# MARK: - HttpxKwargs
class HttpxKwargs(TypedDict, total=False):
    """TypedDict for httpx kwargs."""
    timeout: Union[float, Tuple[float, float], None]
    verify: Union[bool, str]
    cert: Union[str, Tuple[str, str], None]
    proxies: Optional[Dict[str, str]]
    follow_redirects: bool
    params: Optional[Dict[str, Any]]


# MARK: - VectorizeRecord
class VectorizeRecord:
    """Helper class to enforce Cloudflare Vectorize vector format.
    
    Attributes:
        id: Unique identifier for the vector
        text: The original text content
        values: The vector embedding values
        namespace: Optional namespace for the vector
        metadata: Optional metadata associated with the vector
    """

    def __init__(
            self,
            id: str,
            text: str,
            values: List[float],
            namespace: Optional[str] = None,
            metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize a VectorizeRecord.
        
        Args:
            id: Unique identifier for the vector
            text: The original text content
            values: The vector embedding values
            namespace: Optional namespace for the vector
            metadata: Optional metadata associated with the vector
        """
        self.id = id
        self.text = text
        self.values = values
        self.namespace = namespace
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for API requests."""
        vector_dict = {
            "id": self.id,
            "values": self.values,
            "text": self.text,
        }

        if self.namespace:
            vector_dict["namespace"] = self.namespace

        if self.metadata:
            vector_dict["metadata"] = self.metadata

        return vector_dict


# MARK: - CloudflareVectorize
class CloudflareVectorize(VectorStore):
    """Cloudflare Vectorize vector store.

    To use this, you need:
    1. Cloudflare Account ID
    2. Cloudflare API Token with appropriate permissions (Workers AI, Vectorize, D1). Optional if using separate tokens for each service.
    3. Index name (Optional)
    4. D1 Database ID (Optional if using Vectorize only)
    Reference: https://developers.cloudflare.com/api/resources/vectorize/
    """

    def __init__(
            self,
            embedding: Embeddings,
            account_id: str,
            api_token: Optional[str] = None,
            base_url: str = "https://api.cloudflare.com/client/v4",
            d1_database_id: Optional[str] = None,
            index_name: Optional[str] = None,
            **kwargs: Any,
    ) -> None:
        """Initialize a CloudflareVectorize instance.

        Args:
            embedding: Embeddings instance for converting texts to vectors
            account_id: Cloudflare account ID
            api_token: Optional global API token for all Cloudflare services
            base_url: Base URL for Cloudflare API (default: "https://api.cloudflare.com/client/v4")
            d1_database_id: Optional D1 database ID for storing text data
            index_name: Optional name for the default Vectorize index
            **kwargs: Additional arguments including:
                - vectorize_api_token: Token for Vectorize service, if api_token not global scoped
                - d1_api_token: Token for D1 database service, if api_token not global scoped
                - default_wait_seconds: Num of seconds to wait before retrying mutation status check

        Raises:
            ValueError: If required API tokens are not provided
        """
        self.embedding = embedding
        self.account_id = account_id
        self.base_url = base_url
        self.d1_base_url = base_url
        self.d1_database_id = d1_database_id
        self.index_name = index_name
        self.default_wait_seconds = kwargs.get("default_wait_seconds", WAIT_SECONDS)

        # Use the provided API token or get from class level
        self.api_token = api_token
        self.vectorize_api_token = kwargs.get("vectorize_api_token", None)
        self.d1_api_token = kwargs.get("d1_api_token", None)

        # Set headers for Vectorize and D1
        self._headers = {
            "Authorization": f"Bearer {self.vectorize_api_token or self.api_token}",
            "Content-Type": "application/json",
        }
        self.d1_headers = {
            "Authorization": f"Bearer {self.d1_api_token or self.api_token}",
            "Content-Type": "application/json",
        }

        if not self.api_token \
                and not self.vectorize_api_token:
            raise ValueError(
                "Not enough API token values provided.  Please provide a global `api_token` or `vectorize_api_token`.")

        if self.d1_database_id \
                and not self.api_token \
                and not self.d1_api_token:
            raise ValueError(
                "`d1_database_id` provided, but no global `api_token` provided and no `d1_api_token` provided.")

    @property
    def embeddings(self) -> Embeddings:
        """Get the embeddings model used by this vectorstore.

        Returns:
            Embeddings: The embeddings model instance
        """
        return self.embedding

    def _get_url(self, endpoint: str, index_name: str) -> str:
        """Get full URL for an API endpoint.

        Args:
            endpoint: The API endpoint path
            index_name: Name of the Vectorize index

        Returns:
            str: The complete URL for the API endpoint
        """
        return f"{self.base_url}/accounts/{self.account_id}/vectorize/v2/indexes/{index_name}/{endpoint}"

    def _get_base_url(self, endpoint: str) -> str:
        """Get base URL for index management endpoints.
        
        Args:
            endpoint: The API endpoint path

        Returns:
            str: The complete URL for the API endpoint
        """
        return f"{self.base_url}/accounts/{self.account_id}/vectorize/v2/indexes{endpoint}"

    def _get_d1_url(self, endpoint: str) -> str:
        """Get full URL for a D1 API endpoint.
        
        Args:
            endpoint: The API endpoint path

        Returns:
            str: The complete URL for the API endpoint
        """
        return f"{self.d1_base_url}/accounts/{self.account_id}/d1/{endpoint}"

    def _get_d1_base_url(self, endpoint: str) -> str:
        """Get base URL for D1 API endpoints.
        
        Args:
            endpoint: The API endpoint path

        Returns:
            str: The complete URL for the API endpoint
        """

    @staticmethod
    def _get_allowed_kwargs(client_type: str = "requests") -> List[str]:
        """Dynamically determine allowed kwargs for HTTP clients.
        
        Args:
            client_type: HTTP client type ("requests" or "httpx")
            
        Returns:
            List of allowed keyword arguments for the specified client
            
        Raises:
            ValueError: If client_type is not "requests" or "httpx" 
        """
        if client_type == "requests":
            # Get post method signature
            sig = inspect.signature(requests.post)
            # Extract all parameters except the first two (url and data)
            params = list(sig.parameters.keys())[2:]
            # Add common request parameters
            common_params = ["timeout", "verify", "cert", "proxies", "allow_redirects", "stream", "params"]
            return list(set(params + common_params))
        elif client_type == "httpx":
            try:
                import httpx
                # Get post method signature
                sig = inspect.signature(httpx.AsyncClient.post)
                # Extract all parameters except the first two (self and url)
                params = list(sig.parameters.keys())[2:]
                # Add common httpx parameters
                common_params = ["timeout", "verify", "cert", "proxies", "follow_redirects", "params"]
                return list(set(params + common_params))
            except ImportError:
                # Fallback for when httpx is not available
                return ["timeout", "verify", "cert", "proxies", "follow_redirects", "params"]
        else:
            raise ValueError(f"Unsupported client type: {client_type}")

    @staticmethod
    def _filter_request_kwargs(kwargs: Dict[str, Any], client_type: str = "requests") -> Union[
        RequestsKwargs, HttpxKwargs]:
        """Filter kwargs to only include those allowed for the specified HTTP client.
        
        Args:
            kwargs: Dictionary of keyword arguments
            client_type: HTTP client type ("requests" or "httpx")
            
        Returns:
            Dictionary containing only the allowed kwargs for the specified client
            
        Raises:
            ValueError: If client_type is not "requests" or "httpx"
        """
        allowed = CloudflareVectorize._get_allowed_kwargs(client_type)
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed}

        if client_type == "requests":
            return RequestsKwargs(**filtered_kwargs)
        else:
            return HttpxKwargs(**filtered_kwargs)

    # MARK: - _prep_search_request
    @staticmethod
    def _prep_search_request(
            query_embedding: list[float],
            k: int = DEFAULT_TOP_K,
            md_filter: Optional[Dict] = None,
            namespace: Optional[str] = None,
            return_metadata: Optional[str] = "none",
            return_values: bool = False,
    ) -> Dict:
        """Prepare a search request for the Vectorize API.

        Args:
            query_embedding: Vector embedding for the search query
            k: Number of results to return (default: DEFAULT_TOP_K)
            md_filter: Optional metadata filter to apply to results
            namespace: Optional namespace to search within
            return_metadata: Controls metadata return: "none" (default), "indexed", or "all"
            return_values: Whether to return vector values (default: False)

        Returns:
            Dict: Formatted search request for the API
            
        Raises:
            ValueError: If index_name is not provided
        """
        # Prepare search request
        search_request = {
            "vector": query_embedding,
            "topK": k if not return_metadata and not return_values else min(k, DEFAULT_TOP_K_WITH_MD_VALUES),
        }

        if namespace:
            search_request["namespace"] = namespace

        # Add filter if provided
        if md_filter:
            search_request["filter"] = md_filter

        # Add metadata return preference
        if return_metadata:
            search_request["returnMetadata"] = return_metadata

        # Add vector values return preference
        if return_values:
            search_request["returnValues"] = return_values

        return search_request

    # MARK: - _d1_create_upserts
    @staticmethod
    def _d1_create_upserts(table_name: str, data: List[VectorizeRecord], upsert: bool = False) -> List[str]:
        """Create SQL upsert statements for D1 database operations.

        Args:
            table_name: Name of the table to insert/update data in
            data: List of VectorizeRecord objects containing the data

        Returns:
            List[str]: List of SQL statements for upserting the records
            
        Raises:
            ValueError: If table_name is not provided
        """
        statements = []
        for record in data:
            record_dict = record.to_dict()

            if "namespace" not in record_dict.keys():
                record_dict["namespace"] = ""

            if "metadata" not in record_dict.keys():
                record_dict["metadata"] = {}
            else:
                for k, v in record_dict['metadata'].items():
                    record_dict['metadata'][k] = v.replace("'", "''") if isinstance(v, str) else v

            sql = f"INSERT INTO '{table_name}' (id, text, namespace, metadata) " + \
                  f"VALUES (" + \
                  ", ".join(
                      [
                          f"'{x}'"
                          if x else "NULL"
                          for x in
                          [
                              record_dict["id"].replace("'", "''"),
                              record_dict["text"].replace("'", "''"),
                              record_dict["namespace"].replace("'", "''"),
                              json.dumps(record_dict["metadata"])
                          ]
                      ]
                  ) + ")"

            if upsert:
                sql += f"""
                ON CONFLICT (id) DO UPDATE SET
                text = excluded.text,
                namespace = excluded.namespace,
                metadata = excluded.metadata
                """

            statements.append(sql)

        return statements

    # MARK: - _combine_vectorize_and_d1_data
    @staticmethod
    def _combine_vectorize_and_d1_data(
            vector_data: List[Dict[str, Any]],
            d1_response: List[Dict[str, Any]]
    ) -> List[Document]:
        """Combine vector data from Vectorize API with text data from D1 database.

        Args:
            vector_data: List of vector data dictionaries from Vectorize API
            d1_response: Response from D1 database containing text data

        Returns:
            List of Documents with combined data from both sources
            
        Raises:
            ValueError: If index_name is not provided
        """
        # Create a lookup dictionary for D1 text data by ID
        id_to_text = {}
        for item in d1_response:
            if "id" in item and "text" in item:
                id_to_text[item["id"]] = item["text"]

        documents = []
        for vector in vector_data:
            # Create a Document with the complete vector data in metadata
            vector_id = vector.get("id")
            metadata = {}

            # # Add metadata if returned
            if "metadata" in vector:
                metadata.update(vector.get("metadata", {}))

                if "namespace" in vector:
                    metadata["_namespace"] = vector["namespace"]

            # Get the text content from D1 results
            text_content = id_to_text.get(vector_id, "")

            # Add values if returned
            document_args = {
                "id": vector_id,
                "page_content": text_content,
                "metadata": metadata,
            }

            if "values" in vector:
                document_args["values"] = vector.get("values", [])

            documents.append(Document(**document_args))

        return documents

    # MARK: - _poll_mutation_status
    def _poll_mutation_status(
            self,
            index_name: str,
            mutation_id: Optional[str] = None,
            wait_seconds: Optional[int] = None
    ):
        """Poll the mutation status of an index operation until completion.

        Args:
            index_name: Name of the Vectorize index
            mutation_id: Optional ID of the mutation to wait for
            wait_seconds: Optional number of seconds to wait per check interval for mutation status

        Raises:
            Exception: If polling encounters repeated errors
        """
        poll_interval_seconds = wait_seconds or self.default_wait_seconds
        time.sleep(poll_interval_seconds)
        err_cnt = 0
        err_lim = 5
        while True:
            try:
                response_index = self.get_index_info(index_name)
                err_cnt = 0
            except Exception as e:
                if err_cnt >= err_lim:
                    raise Exception("Index Mutation Error:", str(e))
                err_cnt += 1
                time.sleep(2 ** err_cnt)
                continue

            if mutation_id:
                index_mutation_id = response_index.get("processedUpToMutation")
                if index_mutation_id == mutation_id:
                    break
            else:
                return

            time.sleep(poll_interval_seconds)

    # MARK: - _apoll_mutation_status
    async def _apoll_mutation_status(
            self,
            index_name: str,
            mutation_id: Optional[str] = None,
            wait_seconds: Optional[int] = None
    ):
        """Asynchronously poll the mutation status of an index operation until completion.

        Args:
            index_name: Name of the Vectorize index
            mutation_id: Optional ID of the mutation to wait for
            wait_seconds: Optional number of seconds to wait per check interval for mutation status

        Raises:
            Exception: If polling encounters repeated errors
        """
        poll_interval_seconds = wait_seconds or self.default_wait_seconds
        await asyncio.sleep(poll_interval_seconds)
        err_cnt = 0
        err_lim = 5
        while True:
            try:
                response_index = await self.aget_index_info(index_name)
                err_cnt = 0
            except Exception as e:
                if err_cnt >= err_lim:
                    raise Exception("Index Mutation Error:", str(e))
                err_cnt += 1
                await asyncio.sleep(2 ** err_cnt)
                continue

            if mutation_id:
                index_mutation_id = response_index.get("processedUpToMutation")
                if index_mutation_id == mutation_id:
                    break
            else:
                return

            await asyncio.sleep(poll_interval_seconds)

    # MARK: - d1_create_table
    def d1_create_table(self, table_name: str, **kwargs: Any) -> Dict[str, Any]:
        """Create a table in a D1 database using SQL schema.

        Args:
            table_name: Name of the table to create
            **kwargs: Additional keyword arguments to pass to the requests call

        Returns:
            Response data with query results
            
        Raises:
            ValueError: If index_name is not provided
        """

        table_schema = f"""
        CREATE TABLE IF NOT EXISTS '{table_name}' (
            id TEXT PRIMARY KEY, 
            text TEXT, 
            namespace TEXT, 
            metadata TEXT
        )"""

        filtered_kwargs: RequestsKwargs = self._filter_request_kwargs(kwargs, "requests")
        response = requests.post(
            self._get_d1_url(f"database/{self.d1_database_id}/query"),
            headers=self.d1_headers,
            json={"sql": table_schema},
            **filtered_kwargs
        )

        return response.json().get("result", {})

    # MARK: - ad1_create_table
    async def ad1_create_table(self, table_name: str, **kwargs: Any) -> Dict[str, Any]:
        """Asynchronously create a table in a D1 database using SQL schema.

        Args:
            table_name: Name of the table to create
            **kwargs: Additional keyword arguments to pass to the httpx client

        Returns:
            Response data with query results
            
        Raises:
            ValueError: If index_name is not provided
        """

        table_schema = f"""
        CREATE TABLE IF NOT EXISTS '{table_name}' (
            id TEXT PRIMARY KEY, 
            text TEXT, 
            namespace TEXT, 
            metadata TEXT
        )"""

        import httpx

        filtered_kwargs = self._filter_request_kwargs(kwargs, "httpx")
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self._get_d1_url(f"database/{self.d1_database_id}/query"),
                headers=self.d1_headers,
                json={"sql": table_schema},
                **filtered_kwargs
            )
            response.raise_for_status()
            response_data = response.json()

        return response_data.get("result", {})

    # MARK: - d1_drop_table
    def d1_drop_table(self, table_name: str, **kwargs: Any) -> Dict[str, Any]:
        """Asynchronously delete a table from a D1 database.

        Args:
            table_name: Name of the table to delete
            **kwargs: Additional keyword arguments to pass to the requests call

        Returns:
            Response data with query results
            
        Raises:
            ValueError: If index_name is not provided
        """
        drop_query = f"DROP TABLE IF EXISTS '{table_name}'"

        filtered_kwargs: RequestsKwargs = self._filter_request_kwargs(kwargs, "requests")
        response = requests.post(
            self._get_d1_url(f"database/{self.d1_database_id}/query"),
            headers=self.d1_headers,
            json={"sql": drop_query},
            **filtered_kwargs
        )

        response.raise_for_status()

        return response.json().get("result", {})

    # MARK: - ad1_drop_table
    async def ad1_drop_table(self, table_name: str, **kwargs: Any) -> Dict[str, Any]:
        """Asynchronously delete a table from a D1 database.

        Args:
            table_name: Name of the table to delete
            **kwargs: Additional keyword arguments to pass to the httpx client

        Returns:
            Response data with query results
        """
        import httpx

        drop_query = f"DROP TABLE IF EXISTS '{table_name}'"

        filtered_kwargs: HttpxKwargs = self._filter_request_kwargs(kwargs, "httpx")
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self._get_d1_url(f"database/{self.d1_database_id}/query"),
                headers=self.d1_headers,
                json={"sql": drop_query},
                **filtered_kwargs
            )
            response.raise_for_status()
            response_data = response.json()

        return response_data.get("result", {})

    # MARK: - d1_upsert_texts
    def d1_upsert_texts(self, table_name: str, data: List[VectorizeRecord], upsert: bool = False, **kwargs: Any) -> \
            Dict[
                str, Any]:
        """Insert or update text data in a D1 database table.

        Args:
            table_name: Name of the table to insert data into
            data: List of dictionaries containing data to insert
            upsert: If true (default: False), insert or update the text data otherwise insert only.
            **kwargs: Additional keyword arguments to pass to the requests call

        Returns:
            Response data with query results
            
        Raises:
            ValueError: If index_name is not provided
        """
        if not data:
            return {"success": True, "changes": 0}

        statements = self._d1_create_upserts(table_name=table_name, data=data, upsert=upsert)

        filtered_kwargs: RequestsKwargs = self._filter_request_kwargs(kwargs, "requests")
        response = requests.post(
            self._get_d1_url(f"database/{self.d1_database_id}/query"),
            headers=self.d1_headers,
            json={
                "sql": ";\n".join(statements),
            },
            **filtered_kwargs
        )

        return response.json().get("result", {})

    # MARK: - ad1_upsert_texts
    async def ad1_upsert_texts(self, table_name: str, data: List[VectorizeRecord], upsert: bool = False,
                               **kwargs: Any) -> \
            Dict[str, Any]:
        """Asynchronously insert or update text data in a D1 database table.

        Args:
            table_name: Name of the table to insert data into
            data: List of dictionaries containing data to insert
            upsert: If true (default: False), insert or update the text data otherwise insert only.
            **kwargs: Additional keyword arguments to pass to the httpx client

        Returns:
            Response data with query results

        Raises:
            ValueError: If index_name is not provided
        """
        if not data:
            return {"success": True, "changes": 0}

        statements = self._d1_create_upserts(table_name=table_name, data=data, upsert=upsert)

        import httpx

        # Execute with parameters
        filtered_kwargs: HttpxKwargs = self._filter_request_kwargs(kwargs, "httpx")
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self._get_d1_url(f"database/{self.d1_database_id}/query"),
                headers=self.d1_headers,
                json={
                    "sql": ";\n".join(statements),
                },
                **filtered_kwargs
            )
            response.raise_for_status()
            response_data = response.json()

        return response_data.get("result", {})

    # MARK: - d1_get_by_ids
    def d1_get_by_ids(self, table_name: str, ids: List[str], **kwargs: Any) -> List:
        """Retrieve text data from a D1 database table.

        Args:
            table_name: Name of the table to retrieve data from
            ids: List of ids to retrieve
            **kwargs: Additional keyword arguments to pass to the requests call

        Returns:
            Response data with query results
            
        Raises:
            ValueError: If index_name is not provided
        """
        # query D1 for raw results
        placeholders = ','.join(['?'] * len(ids))  # Creates "?,?,?..." for the right number of IDs

        sql = f"""
            SELECT * FROM '{table_name}'
            WHERE id IN ({placeholders})
        """

        filtered_kwargs: RequestsKwargs = self._filter_request_kwargs(kwargs, "requests")
        response = requests.post(
            self._get_d1_url(f"database/{self.d1_database_id}/query"),
            headers=self.d1_headers,
            json={"sql": sql, "params": ids},
            **filtered_kwargs
        )

        response.raise_for_status()
        response_data = response.json()
        d1_results = response_data.get("result", {})
        if len(d1_results) == 0:
            return []

        d1_results_records = d1_results[0].get("results", [])

        return d1_results_records

    # MARK: - ad1_get_texts
    async def ad1_get_by_ids(self, table_name: str, ids: List[str], **kwargs: Any) -> List:
        """Asynchronously retrieve text data from a D1 database table.

        Args:
            table_name: Name of the table to retrieve data from
            ids: List of ids to retrieve
            **kwargs: Additional keyword arguments to pass to the httpx client

        Returns:
            Response data with query results

        Raises:
            ValueError: If index_name is not provided
        """

        # query D1 for raw results
        placeholders = ','.join(['?'] * len(ids))  # Creates "?,?,?..." for the right number of IDs

        sql = f"""
            SELECT * FROM '{table_name}'
            WHERE id IN ({placeholders})
        """

        import httpx

        # Execute the query
        filtered_kwargs: HttpxKwargs = self._filter_request_kwargs(kwargs, "httpx")
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self._get_d1_url(f"database/{self.d1_database_id}/query"),
                headers=self.d1_headers,
                json={
                    "sql": sql,
                    "params": ids,
                },
                **filtered_kwargs
            )
            response.raise_for_status()
            response_data = response.json()

        d1_results = response_data.get("result", {})
        if len(d1_results) == 0:
            return []

        d1_results_records = d1_results[0].get("results", [])

        return d1_results_records

    # MARK: - d1_delete
    def d1_delete(self, table_name: str, ids: List[str], **kwargs) -> Dict[str, Any]:
        """Delete data from a D1 database table.

        Args:
            table_name: Name of the table to delete from
            ids: List of ids to retrieve

        Returns:
            Response data with deletion results

        Raises:
            ValueError: If index_name is not provided
        """
        # query D1 for raw results
        placeholders = ','.join(['?'] * len(ids))  # Creates "?,?,?..." for the right number of IDs

        sql = f"""
                    DELETE FROM '{table_name}'
                    WHERE id IN ({placeholders})
                """
        response = requests.post(
            self._get_d1_url(f"database/{self.d1_database_id}/query"),
            headers=self.d1_headers,
            json={"sql": sql, "params": ids},
            **kwargs
        )
        response.raise_for_status()

        return response.json().get("result", {})

    # MARK: - ad1_delete
    async def ad1_delete(self, table_name: str, ids: List[str], **kwargs: Any) -> Dict[str, Any]:
        """Asynchronously delete data from a D1 database table.

        Args:
            table_name: Name of the table to delete from
            ids: List of ids to retrieve
            **kwargs: Additional keyword arguments to pass to the httpx client

        Returns:
            Response data with deletion results

        Raises:
            ValueError: If index_name is not provided
        """
        # query D1 for raw results
        placeholders = ','.join(['?'] * len(ids))  # Creates "?,?,?..." for the right number of IDs

        sql = f"""
                    DELETE FROM '{table_name}'
                    WHERE id IN ({placeholders})
                """

        import httpx

        # Execute the deletion
        filtered_kwargs: HttpxKwargs = self._filter_request_kwargs(kwargs, "httpx")
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self._get_d1_url(f"database/{self.d1_database_id}/query"),
                headers=self.d1_headers,
                json={
                    "sql": sql,
                    "params": ids,
                },
                **filtered_kwargs
            )
            response.raise_for_status()
            response_data = response.json()

        return response_data.get("result", {})

    # MARK: - add_texts
    def add_texts(
            self,
            texts: Iterable[str],
            metadatas: Optional[List[dict]] = None,
            ids: Optional[List[str]] = None,
            namespaces: Optional[List[str]] = None,
            upsert: bool = False,
            index_name: Optional[str] = None,
            include_d1: bool = True,
            wait: bool = False,
            **kwargs: Any,
    ) -> list[str]:
        """Add texts to the vectorstore.
        
        https://developers.cloudflare.com/api/resources/vectorize/subresources/indexes/methods/insert/
        https://developers.cloudflare.com/api/resources/vectorize/subresources/indexes/methods/upsert/

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of ids to associate with the texts.
            namespaces: Optional list of namespaces for each vector.
            upsert: If True (default: False), uses the insert endpoint which will fail if vectors with
                        the same IDs already exist. If False (default), uses upsert which
                        will create or update vectors.
            index_name: Name of the Vectorize index.
            include_d1: Whether to include D1 database in insert/upsert (default: True)
            wait: If True (default: False), wait until vectors are ready.
            **kwargs: Additional keyword arguments to pass to the requests call

        Returns:
            Operation response with ids of added texts.

        Raises:
            ValueError: If the number of texts exceeds MAX_INSERT_SIZE.
        """
        index_name = index_name or self.index_name

        if not index_name:
            raise ValueError("index_name must be provided")

        # Convert texts to list if it's not already
        texts_list = list(texts)

        # Check if the number of texts exceeds the maximum allowed
        if len(texts_list) > MAX_INSERT_SIZE:
            raise ValueError(
                f"Number of texts ({len(texts_list)}) exceeds maximum allowed ({MAX_INSERT_SIZE})"
            )

        # Generate embeddings for the texts
        embeddings = self.embedding.embed_documents(texts_list)

        # Generate IDs if not provided
        if ids is None or list(set(ids)) == [None]:
            ids = [str(uuid.uuid4()) for _ in texts_list]

        # Prepare vectors with metadata if provided
        vectors = []
        for i, (embedding, id, text) in enumerate(zip(embeddings, ids, texts_list)):
            # Get metadata if provided
            metadata = {}
            if metadatas is not None and i < len(metadatas):
                metadata = metadatas[i].copy()

            # Get namespace if provided
            namespace = None
            if namespaces is not None and i < len(namespaces):
                namespace = namespaces[i]

            # Create VectorizeRecord
            vector = VectorizeRecord(
                id=id,
                text=text,
                values=embedding,
                namespace=namespace,
                metadata=metadata
            )

            vectors.append(vector)

        # Choose endpoint based on upsert parameter
        endpoint = "upsert" if upsert else "insert"

        # Convert vectors to newline-delimited JSON
        vector_dicts = [
            {
                "id": vector.id,
                "values": vector.values,
                "namespace": vector.namespace,
                "metadata": vector.metadata,
            }
            for vector in vectors
        ]
        ndjson_data = "\n".join(json.dumps(x) for x in vector_dicts)

        # Copy headers and set correct content type for NDJSON
        headers = self._headers.copy()
        headers["Content-Type"] = "application/x-ndjson"

        # Make API call to insert/upsert vectors
        filtered_kwargs: RequestsKwargs = self._filter_request_kwargs(kwargs, "requests")
        response = requests.post(
            self._get_url(endpoint, index_name),
            headers=headers,
            data=ndjson_data.encode('utf-8'),
            **filtered_kwargs
        )
        response.raise_for_status()

        if include_d1 and self.d1_database_id:
            self.d1_create_table(
                table_name=index_name,
                **kwargs
            )

            # add values to D1Database
            self.d1_upsert_texts(
                table_name=index_name,
                data=vectors,
                upsert=upsert,
                **kwargs
            )

        mutation_response = response.json()
        mutation_id = mutation_response.get("result", {}).get("mutationId")

        if wait and mutation_id:
            self._poll_mutation_status(
                index_name=index_name,
                mutation_id=mutation_id,
            )

        return ids

    # MARK: - aadd_texts
    async def aadd_texts(
            self,
            texts: Iterable[str],
            metadatas: Optional[List[dict]] = None,
            ids: Optional[List[str]] = None,
            namespaces: Optional[List[str]] = None,
            upsert: bool = False,
            index_name: Optional[str] = None,
            include_d1: bool = True,
            wait: bool = False,
            **kwargs: Any,
    ) -> list[str]:
        """Asynchronously add texts to the vectorstore.
        
        https://developers.cloudflare.com/api/resources/vectorize/subresources/indexes/methods/insert/
        https://developers.cloudflare.com/api/resources/vectorize/subresources/indexes/methods/upsert/
        

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of ids to associate with the texts.
            namespaces: Optional list of namespaces for each vector.
            upsert: If True (default: False), uses the insert endpoint which will fail if vectors with
                        the same IDs already exist. If False (default), uses upsert which
                        will create or update vectors.
            index_name: Name of the Vectorize index.
            include_d1: Whether to include D1 database in insert/upsert (default: True)
            wait: If True (default: False), wait until vectors are ready.

        Returns:
            List of ids from adding the texts into the vectorstore.

        Raises:
            ValueError: If the number of texts exceeds MAX_INSERT_SIZE.
        """
        index_name = index_name or self.index_name

        if not index_name:
            raise ValueError("index_name must be provided")

        # Convert texts to list if it's not already
        texts_list = list(texts)

        # Check if the number of texts exceeds the maximum allowed
        if len(texts_list) > MAX_INSERT_SIZE:
            raise ValueError(
                f"Number of texts ({len(texts_list)}) exceeds maximum allowed ({MAX_INSERT_SIZE})"
            )

        # Generate embeddings for the texts
        embeddings = await self.embedding.aembed_documents(texts_list)

        # Generate IDs if not provided
        if ids is None or list(set(ids)) == [None]:
            ids = [str(uuid.uuid4()) for _ in texts_list]

        # Prepare vectors with metadata if provided
        vectors = []
        for i, (embedding, id, text) in enumerate(zip(embeddings, ids, texts_list)):
            # Get metadata if provided
            metadata = {}
            if metadatas is not None and i < len(metadatas):
                metadata = metadatas[i].copy()

            # Get namespace if provided
            namespace = None
            if namespaces is not None and i < len(namespaces):
                namespace = namespaces[i]

            # Create VectorizeRecord
            vector = VectorizeRecord(
                id=id,
                text=text,
                values=embedding,
                namespace=namespace,
                metadata=metadata
            )

            vectors.append(vector)

        # Choose endpoint based on upsert parameter
        endpoint = "upsert" if upsert else "insert"

        # Convert vectors to newline-delimited JSON
        ndjson_data = "\n".join(json.dumps(vector.to_dict()) for vector in vectors)

        # Import httpx here to avoid dependency issues
        import httpx

        # Copy headers and set correct content type for NDJSON
        headers = self._headers.copy()
        headers["Content-Type"] = "application/x-ndjson"

        # Make API call to insert/upsert vectors
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self._get_url(endpoint, index_name),
                headers=headers,
                content=ndjson_data.encode('utf-8'),
                **kwargs
            )
            response.raise_for_status()

        if include_d1 and self.d1_database_id:
            # create D1 table if not exists
            await self.ad1_create_table(
                table_name=index_name
            )

            # add values to D1Database
            await self.ad1_upsert_texts(
                table_name=index_name,
                data=vectors,
            )

        mutation_response = response.json()
        mutation_id = mutation_response.get("result", {}).get("mutationId")

        if wait and mutation_id:
            await self._apoll_mutation_status(
                index_name=index_name,
                mutation_id=mutation_id,
            )

        return ids

    # MARK: - similarity_search
    def similarity_search(
            self,
            query: str,
            k: int = DEFAULT_TOP_K,
            return_metadata: str = "all",
            index_name: Optional[str] = None,
            md_filter: Optional[Dict[str, Any]] = None,
            namespace: Optional[str] = None,
            return_values: bool = False,
            include_d1: bool = True,
            **kwargs: Any
    ) -> List[Document]:
        """Search for similar documents to a query string.
        
        https://developers.cloudflare.com/api/resources/vectorize/subresources/indexes/methods/query/
        
        Args:
            query: Query string to search for
            k: Number of results to return (default: DEFAULT_TOP_K)
            return_metadata: Controls metadata return: "none", "indexed", or "all" (default: "all")
            index_name: Name of the Vectorize index (optional)
            md_filter: Optional metadata filter to apply to results
            namespace: Optional namespace to search within
            return_values: Whether to return vector embeddings (default: False)
            include_d1: Whether to include D1 database lookups (default: True)
            **kwargs: Additional keyword arguments to pass to the requests call

        Returns:    
            List[Document]: The matching documents
            
            Each Document has:
            - Empty page_content (as Vectorize doesn't store text)
            - metadata containing the complete vector data:
              - id: Identifier for the vector
              - metadata: Any metadata associated with the vector (if requested)
              - page_content: The text content of the document
              - namespace: The namespace the vector belongs to
              - score: The similarity score
              - values: The vector embeddings (if requested)

        Raises:
            ValueError: If index_name is not provided
        """
        index_name = index_name or self.index_name
        if not index_name:
            raise ValueError("index_name must be provided")

        docs_and_scores = \
            self.similarity_search_with_score(
                query=query,
                k=k,
                md_filter=md_filter,
                namespace=namespace,
                index_name=index_name,
                return_metadata=return_metadata,
                return_values=return_values,
                include_d1=include_d1,
                **kwargs
            )

        return docs_and_scores[0]

    # MARK: - similarity_search_with_score
    def similarity_search_with_score(
            self,
            query: str,
            k: int = DEFAULT_TOP_K,
            return_metadata: str = "all",
            return_values: bool = False,
            include_d1: bool = True,
            index_name: Optional[str] = None,
            md_filter: Optional[Dict[str, Any]] = None,
            namespace: Optional[str] = None,
            **kwargs: Any
    ) -> Tuple[List[Document], List[float]]:
        """Search for similar vectors to a query string and return with scores.
        
        https://developers.cloudflare.com/api/resources/vectorize/subresources/indexes/methods/query/

        Args:
            query: Query string to search for
            k: Number of results to return (default: DEFAULT_TOP_K)
            return_metadata: Controls metadata return: "none", "indexed", or "all" (default: "all")
            return_values: Whether to return vector embeddings (default: False)
            include_d1: Whether to include D1 database lookups (default: True)
            index_name: Name of the Vectorize index (optional)
            md_filter: Optional metadata filter to apply to results
            namespace: Optional namespace to search within
            **kwargs: Additional keyword arguments to pass to the requests call

        Returns:
            Tuple of (List of Documents, List of similarity scores).

            Each Document has:
            - Empty page_content (as Vectorize doesn't store text)
            - metadata containing the complete vector data:
              - id: Identifier for the vector
              - metadata: Any metadata associated with the vector (if requested)
              - page_content: The text content of the document
              - namespace: The namespace the vector belongs to
              - score: The similarity score
              - values: The vector embeddings (if requested)

        Raises:
            ValueError: If index_name is not provided
        """
        index_name = index_name or self.index_name

        if not index_name:
            raise ValueError("index_name must be provided")

        # Generate embedding for the query
        query_embedding = self.embedding.embed_query(query)

        search_request = self._prep_search_request(
            query_embedding=query_embedding,
            k=k,
            md_filter=md_filter,
            namespace=namespace,
            return_metadata=return_metadata,
            return_values=return_values
        )

        # Make API call to query vectors
        filtered_kwargs: RequestsKwargs = self._filter_request_kwargs(kwargs, "requests")
        response = requests.post(
            self._get_url("query", index_name),
            headers=self._headers,
            json=search_request,
            **filtered_kwargs
        )
        response.raise_for_status()
        results = response.json().get("result", {}).get("matches", [])

        if include_d1 and self.d1_database_id:
            # query D1 for raw results
            ids = [x.get("id") for x in results]

            d1_results_records = \
                self.d1_get_by_ids(
                    table_name=index_name,
                    ids=ids,
                    **kwargs
                )
        else:
            d1_results_records = []

        # Use _combine_vectorize_and_d1_data to create documents
        documents = self._combine_vectorize_and_d1_data(vector_data=results, d1_response=d1_results_records)

        # Extract scores from results
        scores = [result.get("score", 0.0) for result in results]

        return documents, scores

    # MARK: - asimilarity_search
    async def asimilarity_search(
            self,
            query: str,
            k: int = DEFAULT_TOP_K,
            index_name: Optional[str] = None,
            md_filter: Optional[Dict[str, Any]] = None,
            namespace: Optional[str] = None,
            return_metadata: str = "none",
            return_values: bool = False,
            include_d1: bool = True,
            **kwargs: Any,
    ) -> List[Document]:
        """Asynchronously search for similar documents to a query string.
        
        https://developers.cloudflare.com/api/resources/vectorize/subresources/indexes/methods/query/

        Args:
            query: Query string to search for
            k: Number of results to return (default: DEFAULT_TOP_K)
            return_metadata: Controls metadata return: "none", "indexed", or "all" (default: "all")
            return_values: Whether to return vector embeddings (default: False)
            include_d1: Whether to include D1 database lookups (default: True)
            index_name: Name of the Vectorize index (optional)
            md_filter: Optional metadata filter to apply to results
            namespace: Optional namespace to search within
            **kwargs: Additional arguments to pass to the API call

        Returns:
            List of Documents most similar to the query.
            
            Each Document has:
            - Empty page_content (as Vectorize doesn't store text)
            - metadata containing the complete vector data:
              - id: Identifier for the vector
              - metadata: Any metadata associated with the vector (if requested)
              - page_content: The text content of the document
              - namespace: The namespace the vector belongs to
              - score: The similarity score
              - values: The vector embeddings (if requested)

        Raises:
            ValueError: If index_name is not provided
        """
        index_name = index_name or self.index_name

        if not index_name:
            raise ValueError("index_name must be provided")

        docs_and_scores = \
            await self.asimilarity_search_with_score(
                query=query,
                k=k,
                md_filter=md_filter,
                namespace=namespace,
                index_name=index_name,
                return_metadata=return_metadata,
                return_values=return_values,
                include_d1=include_d1,
                **kwargs
            )
        return docs_and_scores[0]

    # MARK: - asimilarity_search_with_score
    async def asimilarity_search_with_score(
            self,
            query: str,
            k: int = DEFAULT_TOP_K,
            index_name: Optional[str] = None,
            md_filter: Optional[Dict[str, Any]] = None,
            namespace: Optional[str] = None,
            return_metadata: str = "none",
            return_values: bool = False,
            include_d1: bool = True,
            **kwargs: Any,
    ) -> Tuple[List[Document], List[float]]:
        """Asynchronously search for similar vectors to a query string and return with scores.
        
        https://developers.cloudflare.com/api/resources/vectorize/subresources/indexes/methods/query/

        Args:
            query: Query string to search for.
            k: Number of results to return.
            md_filter: Optional metadata filter expression to limit results.
            namespace: Optional namespace to search in.
            index_name: Name of the Vectorize index.
            return_metadata: Controls metadata return: "none" (default), "indexed", or "all".
            return_values: Whether to return vector embeddings (default: False).
            include_d1: Whether to include D1 database lookups (default: True)
            **kwargs: Additional keyword arguments to pass to the httpx client

        Returns:
            Tuple of (List of Documents, List of similarity scores).

            Each Document has:
            - Empty page_content (as Vectorize doesn't store text)
            - metadata containing the complete vector data:
              - id: Identifier for the vector
              - metadata: Any metadata associated with the vector (if requested)
              - page_content: The text content of the document
              - namespace: The namespace the vector belongs to
              - score: The similarity score
              - values: The vector embeddings (if requested)

        Raises:
            ValueError: If index_name is not provided
        """

        index_name = index_name or self.index_name

        if not index_name:
            raise ValueError("index_name must be provided")

        # Generate embedding for the query
        query_embedding = await self.embedding.aembed_query(query)

        search_request = self._prep_search_request(
            query_embedding=query_embedding,
            k=k,
            md_filter=md_filter,
            namespace=namespace,
            return_metadata=return_metadata,
            return_values=return_values
        )

        # Import httpx here to avoid dependency issues
        import httpx

        # Make API call to query vectors
        filtered_kwargs: HttpxKwargs = self._filter_request_kwargs(kwargs, "httpx")
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self._get_url("query", index_name),
                headers=self._headers,
                json=search_request,
                **filtered_kwargs
            )
            response.raise_for_status()
            response_data = response.json()

        results = response_data.get("result", {}).get("matches", [])

        if include_d1 and self.d1_database_id:
            # query D1 for raw results
            ids = [x.get("id") for x in results]

            d1_results_records = \
                await self.ad1_get_by_ids(
                    table_name=index_name,
                    ids=ids,
                    **kwargs
                )
        else:
            d1_results_records = []

        # Use _combine_vectorize_and_d1_data to create documents
        documents = self._combine_vectorize_and_d1_data(results, d1_results_records)

        # Extract scores from results
        scores = [result.get("score", 0.0) for result in results]

        return documents, scores

    # MARK: - delete
    def delete(
            self,
            ids: List[str],
            index_name: Optional[str] = None,
            include_d1: bool = True,
            wait: bool = False,
            **kwargs: Any
    ) -> Dict:
        """Delete vectors by ID from the vectorstore.
        
        https://developers.cloudflare.com/api/resources/vectorize/subresources/indexes/methods/delete_by_ids/

        Args:
            ids: List of ids to delete.
            index_name: Name of the Vectorize index.
            include_d1: Whether to include D1 database lookups (default: True)
            wait: Wait until mutation_id shows that the data has been deleted (default: False).
            
        Returns:
            Dict: The mutation response and ids that were deleted

        Raises:
            ValueError: If index_name is not provided
        """

        index_name = index_name or self.index_name

        if not index_name:
            raise ValueError("index_name must be provided")

        delete_request = {"ids": [str(x) if type(x) != str else x for x in ids]}

        response = requests.post(
            self._get_url("delete_by_ids", index_name),
            headers=self._headers,
            json=delete_request,
            **kwargs
        )
        response.raise_for_status()

        if include_d1 and self.d1_database_id:
            self.d1_delete(
                table_name=index_name,
                ids=ids
            )

        mutation_response = response.json()
        mutation_id = mutation_response.get("result", {}).get("mutationId")

        if wait and mutation_id:
            self._poll_mutation_status(
                index_name=index_name,
                mutation_id=mutation_id,
            )

        return {
            **mutation_response,
            "ids": ids
        }

    # MARK: - adelete
    async def adelete(
            self,
            ids: List[str],
            index_name: Optional[str] = None,
            include_d1: bool = True,
            wait: bool = False,
            **kwargs: Any
    ) -> Dict:
        """Asynchronously delete vectors by ID from the vectorstore.

        https://developers.cloudflare.com/api/resources/vectorize/subresources/indexes/methods/delete_by_ids/

        Args:
            ids: List of ids to delete.
            index_name: Name of the Vectorize index (Optional if passed in class instantiation).
            include_d1: Whether to include D1 database lookups (default: True)
            wait: Wait until mutation_id shows that the data has been deleted (default: False).
            
        Returns:
            Dict: The mutation response and ids that were deleted

        Raises:
            ValueError: If index_name is not provided
        """

        index_name = index_name or self.index_name

        if not index_name:
            raise ValueError("index_name must be provided")

        delete_request = {"ids": [str(x) if type(x) != str else x for x in ids]}

        # Make API call to delete vectors
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self._get_url("delete_by_ids", index_name),
                headers=self._headers,
                json=delete_request,
            )
        response.raise_for_status()

        mutation_response = response.json()
        mutation_id = mutation_response.get("result", {}).get("mutationId")

        if include_d1 and self.d1_database_id:
            await self.ad1_delete(
                table_name=index_name,
                ids=ids,
                **kwargs
            )

        if wait and mutation_id:
            await self._apoll_mutation_status(
                index_name=index_name,
                mutation_id=mutation_id,
            )

        return {
            **mutation_response,
            "ids": ids
        }

    # MARK: - get_by_ids
    def get_by_ids(
            self,
            ids: List[str],
            index_name: Optional[str] = None,
            include_d1: bool = True,
            **kwargs
    ) -> List[Document]:
        """Get vectors by their IDs.

        https://developers.cloudflare.com/api/resources/vectorize/subresources/indexes/methods/get_by_ids/

        Args:
            ids: List of vector IDs to retrieve.
            index_name: Name of the Vectorize index.
            include_d1: Whether to include D1 database lookups (default: True)
            **kwargs: Additional keyword arguments to pass to the requests call

        Returns:
            List[Document]: The matching documents
            
        Raises:
            ValueError: If index_name is not provided
        """

        index_name = index_name or self.index_name

        if not index_name:
            raise ValueError("index_name must be provided")

        get_request = {"ids": [str(x) if type(x) != str else x for x in ids]}

        # Get vector data from Vectorize API
        response = requests.post(
            self._get_url("get_by_ids", index_name),
            headers=self._headers,
            json=get_request,
            **kwargs
        )
        response.raise_for_status()

        vector_data = response.json().get("result", {})

        if include_d1 and self.d1_database_id:
            # Get text data from D1 database
            d1_response = self.d1_get_by_ids(
                table_name=index_name,
                ids=ids,
                **kwargs
            )
        else:
            d1_response = []

        # Combine data into LangChain Document objects
        documents = \
            self._combine_vectorize_and_d1_data(
                vector_data,
                d1_response
            )

        return documents

    # MARK: - aget_by_ids
    async def aget_by_ids(
            self,
            ids: List[str],
            index_name: Optional[str] = None,
            include_d1: bool = True,
            **kwargs
    ) -> List[Document]:
        """Asynchronously get vectors by their IDs.

        https://developers.cloudflare.com/api/resources/vectorize/subresources/indexes/methods/get_by_ids/

        Args:
            ids: List of vector IDs to retrieve.
            index_name: Name of the Vectorize index.
            include_d1: Whether to include D1 database lookups (default: True)
            **kwargs: Additional keyword arguments to pass to the requests call

        Returns:
            List[Document]: The matching documents
            
        Raises:
            ValueError: If index_name is not provided
        """

        index_name = index_name or self.index_name

        if not index_name:
            raise ValueError("index_name must be provided")

        get_request = {"ids": [str(x) if type(x) != str else x for x in ids]}

        # Get vector data from Vectorize API
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self._get_url("get_by_ids", index_name),
                headers=self._headers,
                json=get_request,
                **kwargs
            )
        response.raise_for_status()

        vector_data = response.json().get("result", {})

        if include_d1 and self.d1_database_id:
            d1_response = await self.ad1_get_by_ids(
                table_name=index_name,
                ids=ids,
                **kwargs
            )
        else:
            d1_response = []

        documents = \
            self._combine_vectorize_and_d1_data(
                vector_data,
                d1_response
            )

        return documents

    # MARK: - get_index_info
    def get_index_info(self, index_name: str, **kwargs) -> Dict[str, Any]:
        """Get information about the current index.
        
        https://developers.cloudflare.com/api/resources/vectorize/subresources/indexes/methods/info/
        
        Args:
            index_name: Name of the Vectorize index.
            **kwargs: Additional keyword arguments to pass to the requests call

        Returns:
            Dictionary containing index information.
            
        Raises:
            ValueError: If index_name is not provided
        """

        index_name = index_name or self.index_name

        if not index_name:
            raise ValueError("index_name must be provided")

        response = requests.get(
            self._get_url("info", index_name),
            headers=self._headers,
            **kwargs
        )
        response.raise_for_status()

        return response.json().get("result", {})

    # MARK: - aget_index_info
    async def aget_index_info(self, index_name: str, **kwargs) -> Dict[str, Any]:
        """Asynchronously get information about the current index.
        
        https://developers.cloudflare.com/api/resources/vectorize/subresources/indexes/methods/info/
        
        Args:
            index_name: Name of the Vectorize index.
            **kwargs: Additional keyword arguments to pass to the httpx call

        Returns:
            Dictionary containing index information.
            
        Raises:
            ValueError: If index_name is not provided
        """

        index_name = index_name or self.index_name

        if not index_name:
            raise ValueError("index_name must be provided")

        # Import httpx here to avoid dependency issues
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.get(
                self._get_url("info", index_name),
                headers=self._headers,
                **kwargs
            )
            response.raise_for_status()
            response_data = response.json()

        return response_data.get("result", {})

    # MARK: - create_metadata_index
    def create_metadata_index(
            self, property_name: str,
            index_type: str = "string",
            index_name: Optional[str] = None,
            wait: bool = False,
            **kwargs) -> Dict:
        """Create a metadata index for a specific property.
        
        https://developers.cloudflare.com/api/resources/vectorize/subresources/indexes/subresources/metadata_index/methods/create/

        Args:
            property_name: The metadata property to index.
            index_type: The type of index to create (default: "string").
            index_name: Name of the Vectorize index.
            wait: Wait until mutation_id shows that the index is created (default: False).

        Returns:
            Response data with mutation ID.
            
        Raises:
            ValueError: If index_name is not provided
        """

        index_name = index_name or self.index_name

        if not index_name:
            raise ValueError("index_name must be provided")

        response = requests.post(
            f"{self._get_url('metadata_index/create', index_name)}",
            headers=self._headers,
            json={
                "propertyName": property_name,
                "indexType": index_type
            },
            **kwargs
        )
        response.raise_for_status()
        mutation_response = response.json()
        mutation_id = mutation_response.get("result", {}).get("mutationId")

        if wait:
            self._poll_mutation_status(index_name=index_name, mutation_id=mutation_id)

        return response.json().get("result", {})

    # MARK: - acreate_metadata_index
    async def acreate_metadata_index(
            self,
            property_name: str,
            index_type: str = "string",
            index_name: Optional[str] = None,
            wait: bool = False,
            **kwargs) -> Dict:
        """Asynchronously create a metadata index for a specific property.
        
        https://developers.cloudflare.com/api/resources/vectorize/subresources/indexes/subresources/metadata_index/methods/create/

        Args:
            property_name: The metadata property to index.
            index_type: The type of index to create (default: "string").
            index_name: Name of the Vectorize index.
            wait: Wait until the index is created (default: False).

        Returns:
            Response data with mutation ID.
            
        Raises:
            ValueError: If index_name is not provided
        """

        index_name = index_name or self.index_name

        if not index_name:
            raise ValueError("index_name must be provided")

        # Import httpx here to avoid dependency issues
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self._get_url('metadata_index/create', index_name)}",
                headers=self._headers,
                json={
                    "propertyName": property_name,
                    "indexType": index_type
                },
                **kwargs
            )
            response.raise_for_status()

            mutation_response = response.json()
            mutation_id = mutation_response.get("result", {}).get("mutationId")

        if wait:
            await self._apoll_mutation_status(index_name=index_name, mutation_id=mutation_id)

        return response.json().get("result", {})

    # MARK: - list_metadata_indexes
    def list_metadata_indexes(
            self,
            index_name: Optional[str] = None,
            **kwargs
    ) -> List[Dict[str, str]]:
        """List all metadata indexes for the current index.
        
        https://developers.cloudflare.com/api/resources/vectorize/subresources/indexes/subresources/metadata_index/methods/list/

        Args:
            index_name: Name of the Vectorize index.
            **kwargs: Additional keyword arguments to pass to the requests call

        Returns:
            List of metadata indexes with their property names and index types.
            
        Raises:
            ValueError: If index_name is not provided
        """

        index_name = index_name or self.index_name

        if not index_name:
            raise ValueError("index_name must be provided")

        response = requests.get(
            f"{self._get_url('metadata_index/list', index_name)}",
            headers=self._headers,
            **kwargs
        )
        response.raise_for_status()

        return response.json().get("result", {}).get("metadataIndexes", [])

    # MARK: - alist_metadata_indexes
    async def alist_metadata_indexes(
            self,
            index_name: Optional[str] = None,
            **kwargs
    ) -> List[Dict[str, str]]:
        """Asynchronously list all metadata indexes for the current index.
        
        https://developers.cloudflare.com/api/resources/vectorize/subresources/indexes/subresources/metadata_index/methods/list/

        Args:
            index_name: Name of the Vectorize index.
            **kwargs: Additional keyword arguments to pass to the requests call

        Returns:
            List of metadata indexes with their property names and index types.
            
        Raises:
            ValueError: If index_name is not provided
        """

        index_name = index_name or self.index_name

        if not index_name:
            raise ValueError("index_name must be provided")

        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self._get_url('metadata_index/list', index_name)}",
                headers=self._headers,
                **kwargs
            )
            response.raise_for_status()

            response_data = response.json()

        return response_data.get("result", {}).get("metadataIndexes", [])

    # MARK: - delete_metadata_index
    def delete_metadata_index(
            self,
            property_name: str,
            index_name: Optional[str] = None,
            **kwargs
    ) -> Dict[str, Any]:
        """Delete a metadata index.
        
        https://developers.cloudflare.com/api/resources/vectorize/subresources/indexes/subresources/metadata_index/methods/delete/

        Args:
            property_name: The metadata property index to delete.
            index_name: Name of the Vectorize index.

        Returns:
            Response data with mutation ID.
            
        Raises:
            ValueError: If index_name is not provided
        """

        index_name = index_name or self.index_name

        if not index_name:
            raise ValueError("index_name must be provided")

        response = requests.post(
            f"{self._get_url('metadata_index/delete', index_name)}",
            headers=self._headers,
            json={"property": property_name},
            **kwargs
        )
        response.raise_for_status()
        return response.json().get("result", {})

    # MARK: - adelete_metadata_index
    async def adelete_metadata_index(
            self,
            property_name: str,
            index_name: Optional[str] = None,
            **kwargs
    ) -> Dict[str, Any]:
        """Asynchronously delete a metadata index.
        
        https://developers.cloudflare.com/api/resources/vectorize/subresources/indexes/subresources/metadata_index/methods/delete/

        Args:
            property_name: The metadata property to remove indexing for.
            index_name: Name of the Vectorize index.

        Returns:
            Response data with mutation ID.
            
        Raises:
            ValueError: If index_name is not provided
        """

        index_name = index_name or self.index_name

        if not index_name:
            raise ValueError("index_name must be provided")

        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self._get_url('metadata_index/delete', index_name)}",
                headers=self._headers,
                json={"propertyName": property_name},
                **kwargs
            )
            response.raise_for_status()

            response_data = response.json()

        return response_data.get("result", {})

    # MARK: - get_index
    def get_index(
            self,
            index_name: Optional[str] = None,
            **kwargs
    ) -> Dict[str, Any]:
        """Get information about the current Vectorize index.

        https://developers.cloudflare.com/api/resources/vectorize/subresources/indexes/methods/get/
        
        Args:
            index_name: Name of the Vectorize index.
            **kwargs: Additional keyword arguments to pass to the requests call

        Returns:
            Dictionary containing index configuration and details.
            
        Raises:
            ValueError: If index_name is not provided
        """

        index_name = index_name or self.index_name

        if not index_name:
            raise ValueError("index_name must be provided")

        response = requests.get(
            f"{self.base_url}/accounts/{self.account_id}/vectorize/v2/indexes/{index_name}",
            headers=self._headers,
            **kwargs
        )
        response.raise_for_status()

        return response.json().get("result", {})

    # MARK: - aget_index
    async def aget_index(
            self,
            index_name: Optional[str] = None,
            **kwargs
    ) -> Dict[str, Any]:
        """Asynchronously get information about the current Vectorize index.

        https://developers.cloudflare.com/api/resources/vectorize/subresources/indexes/methods/get/

        Args:
            index_name: Name of the Vectorize index.
            **kwargs: Additional keyword arguments to pass to the requests call

        Returns:
            Dictionary containing index information.
            
        Raises:
            ValueError: If index_name is not provided
        """

        index_name = index_name or self.index_name

        if not index_name:
            raise ValueError("index_name must be provided")

        # Import httpx here to avoid dependency issues
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/accounts/{self.account_id}/vectorize/v2/indexes/{index_name}",
                headers=self._headers,
                **kwargs
            )
            response.raise_for_status()

            response_data = response.json()

        return response_data.get("result", {})

    # MARK: - create_index
    def create_index(
            self,
            dimensions: int = DEFAULT_DIMENSIONS,
            metric: str = DEFAULT_METRIC,
            index_name: Optional[str] = None,
            description: Optional[str] = None,
            include_d1: bool = True,
            wait: bool = False,
            **kwargs
    ) -> Dict[str, Any]:
        """Create a new Vectorize index.
        
        https://developers.cloudflare.com/api/resources/vectorize/subresources/indexes/methods/create/

        Args:
            dimensions: Number of dimensions for the vector embeddings
            metric: Distance metric to use (e.g., "cosine", "euclidean")
            index_name: Name for the new index
            description: Optional description for the index
            include_d1: Whether to create a D1 table for the index
            wait: Whether to wait for the index to be created
            **kwargs: Additional keyword arguments to pass to the requests call

        Returns:
            Response data from the API
            
        Raises:
            ValueError: If index_name is not provided
        """
        index_name = index_name or self.index_name

        # Use provided token or get class level token
        token = self.vectorize_api_token or self.api_token

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        data = {
            "name": index_name,
            "config": {
                "dimensions": dimensions,
                "metric": metric,
            }
        }

        if description:
            data["description"] = description

        # Check if index already exists - but handle various error states
        try:
            r = self.get_index(index_name, **kwargs)
            if r:
                raise ValueError(f"Index {index_name} already exists")
        except requests.exceptions.HTTPError as e:
            # If 404 Not Found or 410 Gone, we can create the index
            if e.response.status_code in [404, 410]:
                pass  # Index doesn't exist or was removed, so we can create it
            else:
                # Re-raise for other HTTP errors
                raise

        # Create the index
        response = requests.post(
            f"{self.base_url}/accounts/{self.account_id}/vectorize/v2/indexes",
            headers=headers,
            json=data,
            **kwargs
        )
        response.raise_for_status()

        if include_d1 and self.d1_database_id:
            # Create D1 table if not exists
            self.d1_create_table(
                table_name=index_name,
                **kwargs
            )

        if wait:
            self._poll_mutation_status(index_name=index_name)

        return response.json().get("result", {})

    # MARK: - acreate_index
    async def acreate_index(
            self,
            dimensions: int = DEFAULT_DIMENSIONS,
            metric: str = DEFAULT_METRIC,
            index_name: Optional[str] = None,
            description: Optional[str] = None,
            include_d1: bool = True,
            wait: bool = False,
            **kwargs
    ) -> Dict[str, Any]:
        """Asyncronously Create a new Vectorize index.
        
        https://developers.cloudflare.com/api/resources/vectorize/subresources/indexes/methods/create/

        Args:
            dimensions: Number of dimensions for the vector embeddings
            metric: Distance metric to use (e.g., "cosine", "euclidean")
            index_name: Name for the new index
            description: Optional description for the index
            include_d1: Whether to create a D1 table for the index
            wait: Whether to wait for the index to be created
            **kwargs: Additional keyword arguments to pass to the requests call

        Returns:
            Response data from the API
            
        Raises:
            ValueError: If index_name is not provided
        """
        index_name = index_name or self.index_name

        # Use provided token or get class level token
        token = self.vectorize_api_token or self.api_token

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        data = {
            "name": index_name,
            "config": {
                "dimensions": dimensions,
                "metric": metric,
            }
        }

        if description:
            data["description"] = description

        # Import httpx here to avoid dependency issues
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/accounts/{self.account_id}/vectorize/v2/indexes",
                headers=headers,
                json=data,
                **kwargs
            )
            response.raise_for_status()

            response_data = response.json()

        if include_d1 and self.d1_database_id:
            # create D1 table if not exists
            await self.ad1_create_table(
                table_name=index_name,
                **kwargs
            )

        if wait:
            await self._apoll_mutation_status(index_name=index_name)

        return response_data.get("result", {})

    # MARK: - list_indexes
    def list_indexes(
            self,
            **kwargs
    ) -> List[Dict[str, Any]]:
        """List all Vectorize indexes for the account.
        
        https://developers.cloudflare.com/api/resources/vectorize/subresources/indexes/methods/list/

        Args:
            **kwargs: Additional arguments to pass to the API request

        Returns:
            List[Dict[str, Any]]: List of index configurations and details
            
        Raises:
            ValueError: If index_name is not provided
        """
        # Use provided token or get class level token
        token = self.api_token or self.vectorize_api_token

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        response = requests.get(
            f"{self.base_url}/accounts/{self.account_id}/vectorize/v2/indexes",
            headers=headers,
            **kwargs
        )
        response.raise_for_status()

        return response.json().get("result", [])

    # MARK: - alist_indexes
    async def alist_indexes(
            self,
            **kwargs
    ) -> List[Dict[str, Any]]:
        """Asynchronously list all Vectorize indexes for the account.
        
        https://developers.cloudflare.com/api/resources/vectorize/subresources/indexes/methods/list/

        Args:
            **kwargs: Additional arguments to pass to the httpx request

        Returns:
            List[Dict[str, Any]]: List of index configurations and details
            
        Raises:
            ValueError: If index_name is not provided
        """
        # Use provided token or get class level token
        token = self.vectorize_api_token or self.api_token

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        # Import httpx here to avoid dependency issues
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/accounts/{self.account_id}/vectorize/v2/indexes",
                headers=headers,
                **kwargs
            )
            response.raise_for_status()

            response_data = response.json()

        return response_data.get("result", [])

    # MARK: - delete_index
    def delete_index(
            self,
            index_name: Optional[str] = None,
            include_d1: bool = True,
            **kwargs
    ) -> Dict[str, Any]:
        """Delete a Vectorize index and optionally its associated D1 table.
        
        https://developers.cloudflare.com/api/resources/vectorize/subresources/indexes/methods/delete/

        Args:
            index_name: Name of the index to delete (uses instance default if not provided)
            include_d1: Whether to also delete the associated D1 table (default: True)
            **kwargs: Additional arguments to pass to the requests call

        Returns:
            Dict[str, Any]: Response data from the deletion operation

        Raises:
            ValueError: If index_name is not provided and no default exists
        """
        index_name = index_name or self.index_name

        # Use provided token or get class level token
        token = self.vectorize_api_token or self.api_token

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        response = requests.delete(
            f"{self.base_url}/accounts/{self.account_id}/vectorize/v2/indexes/{index_name}",
            headers=headers,
            **kwargs
        )
        response.raise_for_status()

        if include_d1 and self.d1_database_id:
            # delete D1 table if exists
            self.d1_drop_table(
                table_name=index_name
            )

        return response.json().get("result", {})

    # MARK: - adelete_index
    async def adelete_index(
            self,
            index_name: Optional[str] = None,
            include_d1: bool = True,
            **kwargs
    ) -> Dict[str, Any]:
        """Asynchronously delete a Vectorize index.
        
        https://developers.cloudflare.com/api/resources/vectorize/subresources/indexes/methods/delete/

        Args:
            index_name: Name of the index to delete (uses instance default if not provided)
            include_d1: Whether to also delete the associated D1 table (default: True)
            **kwargs: Additional arguments to pass to the httpx call

        Returns:
            Dict[str, Any]: Response data from the deletion operation

        Raises:
            ValueError: If index_name is not provided and no default exists
        """
        index_name = index_name or self.index_name

        # Use provided token or get class level token
        token = self.vectorize_api_token or self.api_token

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        # Import httpx here to avoid dependency issues
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.delete(
                f"{self.base_url}/accounts/{self.account_id}/vectorize/v2/indexes/{index_name}",
                headers=headers,
                **kwargs
            )
            response.raise_for_status()

            response_data = response.json()

        if include_d1 and self.d1_database_id:
            await self.ad1_drop_table(
                table_name=index_name
            )

        return response_data.get("result", {})

    # MARK: - from_texts
    @classmethod
    def from_texts(
            cls: Type[VST],
            texts: List[str],
            embedding: Embeddings,
            metadatas: Optional[List[dict]] = None,
            ids: Optional[List[str]] = None,
            namespaces: Optional[List[str]] = None,
            upsert: bool = False,
            account_id: Optional[str] = None,
            d1_database_id: Optional[str] = None,
            index_name: Optional[str] = None,
            dimensions: int = DEFAULT_DIMENSIONS,
            metric: str = DEFAULT_METRIC,
            api_token: Optional[str] = None,
            **kwargs: Any,
    ) -> VST:
        """Create a CloudflareVectorize vectorstore from raw texts.
        
        https://developers.cloudflare.com/api/resources/vectorize/subresources/indexes/methods/create/

        Args:
            texts: List of texts to add to the vectorstore
            embedding: Embedding function to use for the vectors
            metadatas: List of metadata dictionaries to associate with the texts
            ids: List of unique identifiers for the texts
            namespaces: List of namespaces for the texts
            upsert: Whether to upsert the texts into the vectorstore
            account_id: Cloudflare account ID
            d1_database_id: D1 database ID
            index_name: Name for the new index
            dimensions: Number of dimensions for the vector embeddings
            metric: Distance metric to use (e.g., "cosine", "euclidean", default: "cosine")
            api_token: Cloudflare API token, optional if using separate tokens for each service
            **kwargs: Additional keyword arguments to pass to the requests call

        Returns:
            CloudflareVectorize: A new CloudflareVectorize vectorstore

        Raises:
            ValueError: If account_id or index_name is not provided
        """
        index_name = index_name or cls.index_name

        if not account_id or not index_name:
            raise ValueError("account_id and index_name must be provided")

        if not ids:
            ids = [uuid.uuid4().hex for _ in range(len(texts))]

        vectorize_api_token = kwargs.pop("vectorize_api_token", None)
        d1_api_token = kwargs.pop("d1_api_token", None)

        vectorstore = cls(
            embedding=embedding,
            account_id=account_id,
            d1_database_id=d1_database_id,
            api_token=api_token,
            vectorize_api_token=vectorize_api_token,
            d1_api_token=d1_api_token
        )

        # create vectorize index if not exists
        vectorstore.create_index(
            index_name=index_name,
            dimensions=dimensions,
            metric=metric,
            **kwargs
        )

        if kwargs.get("wait"):
            vectorstore._poll_mutation_status(index_name=index_name)

        vectorstore.add_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids,
            namespaces=namespaces,
            upsert=upsert,
            index_name=index_name,
            **kwargs,
        )

        return vectorstore

    # MARK: - afrom_texts
    @classmethod
    async def afrom_texts(
            cls: Type[VST],
            texts: List[str],
            embedding: Embeddings,
            metadatas: Optional[List[dict]] = None,
            ids: Optional[List[str]] = None,
            namespaces: Optional[List[str]] = None,
            upsert: bool = False,
            account_id: Optional[str] = None,
            d1_database_id: Optional[str] = None,
            index_name: Optional[str] = None,
            dimensions: int = DEFAULT_DIMENSIONS,
            metric: str = DEFAULT_METRIC,
            api_token: Optional[str] = None,
            **kwargs: Any,
    ) -> VST:
        """Asynchronously create a CloudflareVectorize vectorstore from raw texts.
        
        https://developers.cloudflare.com/api/resources/vectorize/subresources/indexes/methods/create/

        Args:
            texts: List of texts to add to the vectorstore
            embedding: Embedding function to use for the vectors
            metadatas: List of metadata dictionaries to associate with the texts
            ids: List of unique identifiers for the texts
            namespaces: List of namespaces for the texts
            upsert: Whether to upsert the texts into the vectorstore
            account_id: Cloudflare account ID
            d1_database_id: D1 database ID
            index_name: Name for the new index
            dimensions: Number of dimensions for the vector embeddings
            metric: Distance metric to use (e.g., "cosine", "euclidean", default: "cosine")
            api_token: Cloudflare API token, optional if using separate tokens for each service
            **kwargs: Additional keyword arguments to pass to the httpx call

        Returns:
            CloudflareVectorize: A new CloudflareVectorize vectorstore

        Raises:
            ValueError: If account_id or index_name is not provided
        """

        index_name = index_name or cls.index_name

        if not account_id or not index_name:
            raise ValueError("account_id and index_name must be provided")

        if not ids:
            ids = [uuid.uuid4().hex for _ in range(len(texts))]

        vectorize_api_token = kwargs.pop("vectorize_api_token", None)
        d1_api_token = kwargs.pop("d1_api_token", None)

        vectorstore = cls(
            embedding=embedding,
            account_id=account_id,
            d1_database_id=d1_database_id,
            api_token=api_token,
            vectorize_api_token=vectorize_api_token,
            d1_api_token=d1_api_token
        )

        await vectorstore.acreate_index(
            index_name=index_name,
            dimensions=dimensions,
            metric=metric,
            **kwargs
        )

        if kwargs.get("wait"):
            await vectorstore._apoll_mutation_status(index_name=index_name)

        await vectorstore.aadd_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids,
            namespaces=namespaces,
            upsert=upsert,
            index_name=index_name,
            **kwargs
        )

        return vectorstore

    # MARK: - from_documents
    @classmethod
    def from_documents(
            cls: Type[VST],
            documents: List[Document],
            embedding: Embeddings,
            namespaces: Optional[List[str]] = None,
            upsert: bool = False,
            account_id: Optional[str] = None,
            d1_database_id: Optional[str] = None,
            index_name: Optional[str] = None,
            dimensions: int = DEFAULT_DIMENSIONS,
            metric: str = DEFAULT_METRIC,
            api_token: Optional[str] = None,
            **kwargs: Any,
    ) -> VST:
        """Create a CloudflareVectorize vectorstore from documents.
        
            https://developers.cloudflare.com/api/resources/vectorize/subresources/indexes/methods/create/

        Args:
            documents: List of Documents to add to the vectorstore.
            embedding: Embedding function to use to embed the documents.
            ids: Optional list of ids to associate with the documents.
            namespaces: Optional list of namespaces for each vector.
            upsert: If True (default: False), uses insert instead of upsert.
            account_id: Cloudflare account ID.
            d1_database_id: D1 database ID (Optional if using Vectorize only).
            index_name: Name of the Vectorize index.
            dimensions: Number of dimensions for vectors when creating a new index.
            metric: Distance metric to use when creating a new index.
            api_token: Cloudflare API token, optional if using separate tokens for each service.
            **kwargs: Additional keyword arguments to pass to the requests call

        Returns:
            CloudflareVectorize vectorstore.

        Raises:
            ValueError: If the number of documents exceeds MAX_INSERT_SIZE.
        """
        index_name = index_name or cls.index_name

        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        if "ids" not in kwargs:
            ids = [doc.id for doc in documents]
        else:
            ids = None

        return cls.from_texts(
            texts=texts,
            embedding=embedding,
            metadatas=metadatas,
            ids=ids,
            namespaces=namespaces,
            upsert=upsert,
            account_id=account_id,
            d1_database_id=d1_database_id,
            index_name=index_name,
            dimensions=dimensions,
            metric=metric,
            api_token=api_token,
            **kwargs,
        )

    # MARK: - afrom_documents
    @classmethod
    async def afrom_documents(
            cls: Type[VST],
            documents: List[Document],
            embedding: Embeddings,
            namespaces: Optional[List[str]] = None,
            upsert: bool = False,
            account_id: Optional[str] = None,
            d1_database_id: Optional[str] = None,
            index_name: Optional[str] = None,
            dimensions: int = DEFAULT_DIMENSIONS,
            metric: str = DEFAULT_METRIC,
            api_token: Optional[str] = None,
            **kwargs: Any,
    ) -> VST:
        """Asynchronously create a CloudflareVectorize vectorstore from documents.
        
        https://developers.cloudflare.com/api/resources/vectorize/subresources/indexes/methods/create/

        Args:
            documents: List of Documents to add to the vectorstore.
            embedding: Embedding function to use to embed the documents.
            ids: Optional list of ids to associate with the documents.
            namespaces: Optional list of namespaces for each vector.
            upsert: If True (default: False), uses insert instead of upsert.
            account_id: Cloudflare account ID.
            d1_database_id: D1 database ID (Optional if using Vectorize only).
            index_name: Name of the Vectorize index.
            dimensions: Number of dimensions for vectors when creating a new index.
            metric: Distance metric to use when creating a new index.
            api_token: Cloudflare API token, optional if using separate tokens for each service.
            **kwargs: Additional keyword arguments to pass to the httpx call

        Returns:
            CloudflareVectorize vectorstore.

        Raises:
            ValueError: If the number of documents exceeds MAX_INSERT_SIZE.
        """
        index_name = index_name or cls.index_name

        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        if "ids" not in kwargs:
            ids = [doc.id for doc in documents]
        else:
            ids = None

        return await cls.afrom_texts(
            texts=texts,
            embedding=embedding,
            metadatas=metadatas,
            ids=ids,
            namespaces=namespaces,
            upsert=upsert,
            account_id=account_id,
            d1_database_id=d1_database_id,
            index_name=index_name,
            dimensions=dimensions,
            metric=metric,
            api_token=api_token,
            **kwargs,
        )

    # MARK: - add_documents
    def add_documents(
            self,
            documents: List[Document],
            namespaces: Optional[List[str]] = None,
            upsert: bool = False,
            index_name: Optional[str] = None,
            wait: bool = False,
            **kwargs: Any,
    ) -> list[str]:
        """Add documents to the vectorstore.
        
        https://developers.cloudflare.com/api/resources/vectorize/subresources/indexes/methods/insert/

        Args:
            documents: List of Documents to add to the vectorstore.
            namespaces: Optional list of namespaces for each vector.
            upsert: If True (default: False), uses the insert endpoint which will fail if vectors with
                        the same IDs already exist. If False (default), uses upsert which
                        will create or update vectors.
            index_name: Name of the Vectorize index.
            wait: If True (default: False), poll until all documents have been added.
            **kwargs: Additional keyword arguments to pass to the requests call

        Returns:
            Operation response with ids of added documents.

        Raises:
            ValueError: If the number of documents exceeds MAX_INSERT_SIZE.
        """
        index_name = index_name or self.index_name

        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        # Only generate and pass ids if not already in kwargs
        if "ids" not in kwargs:
            ids = [doc.id or str(uuid.uuid4()) for doc in documents]
            kwargs["ids"] = ids

        return self.add_texts(
            texts=texts,
            metadatas=metadatas,
            namespaces=namespaces,
            upsert=upsert,
            index_name=index_name,
            wait=wait,
            **kwargs
        )

    # MARK: - aadd_documents
    async def aadd_documents(
            self,
            documents: List[Document],
            ids: Optional[List[str]] = None,
            namespaces: Optional[List[str]] = None,
            upsert: bool = False,
            index_name: Optional[str] = None,
            include_d1: bool = True,
            **kwargs: Any,
    ) -> list[str]:
        """Asynchronously add documents to the vectorstore.
        
        https://developers.cloudflare.com/api/resources/vectorize/subresources/indexes/methods/insert/

        Args:
            documents: List of Documents to add to the vectorstore.
            ids: Optional list of ids to associate with the documents.
            namespaces: Optional list of namespaces for each vector.
            upsert: If True (default: False), uses the insert endpoint which will fail if vectors with
                        the same IDs already exist. If False (default), uses upsert which
                        will create or update vectors.
            index_name: Name of the Vectorize index.
            include_d1: Whether to also add the documents to the D1 table (default: True)
            **kwargs: Additional keyword arguments to pass to the httpx call
            
        Returns:
            List of ids from adding the documents into the vectorstore.

        Raises:
            ValueError: If the number of documents exceeds MAX_INSERT_SIZE.
        """
        index_name = index_name or self.index_name

        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        # Use document IDs if they exist, falling back to provided ids
        if "ids" not in kwargs:
            ids = [doc.id for doc in documents]

        return await self.aadd_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids,
            namespaces=namespaces,
            upsert=upsert,
            index_name=index_name,
            include_d1=include_d1,
            **kwargs
        )
