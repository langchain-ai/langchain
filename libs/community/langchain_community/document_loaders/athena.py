from __future__ import annotations

import time
from typing import Any, Dict, Iterator, List, Optional, Tuple

from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader


class AthenaLoader(BaseLoader):
    """Load documents from `AWS Athena`.

    Each document represents one row of the result.
    By default, all columns are written into the `page_content` and none into the `metadata`.
    If `metadata_columns` are provided then these columns are written into the `metadata` of the
    document while the rest of the columns are written into the `page_content` of the document.

    To authenticate, the AWS client uses the following methods to automatically load credentials:
    https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html

    If a specific credential profile should be used, you must pass
    the name of the profile from the ~/.aws/credentials file that is to be used.

    Make sure the credentials / roles used have the required policies to
    access the Amazon Textract service.
    """

    def __init__(
        self,
        query: str,
        database: str,
        s3_output_uri: str,
        profile_name: str,
        metadata_columns: Optional[List[str]] = None,
    ):
        """Initialize Athena document loader.

        Args:
            query: The query to run in Athena.
            database: Athena database
            s3_output_uri: Athena output path
            metadata_columns: Optional. Columns written to Document `metadata`.
        """
        self.query = query
        self.database = database
        self.s3_output_uri = s3_output_uri
        self.profile_name = profile_name
        self.metadata_columns = metadata_columns if metadata_columns is not None else []

    def _execute_query(self) -> List[Dict[str, Any]]:
        import boto3

        session = (
            boto3.Session(profile_name=self.profile_name)
            if self.profile_name is not None
            else boto3.Session()
        )
        client = session.client("athena")

        response = client.start_query_execution(
            QueryString=self.query,
            QueryExecutionContext={"Database": self.database},
            ResultConfiguration={"OutputLocation": self.s3_output_uri},
        )
        query_execution_id = response["QueryExecutionId"]

        while True:
            response = client.get_query_execution(QueryExecutionId=query_execution_id)
            state = response["QueryExecution"]["Status"]["State"]
            if state == "SUCCEEDED":
                break
            elif state == "FAILED":
                raise Exception(
                    f"Query Failed: {response['QueryExecution']['Status']['StateChangeReason']}"
                )
            elif state == "CANCELLED":
                raise Exception("Query was cancelled by the user.")
            else:
                print(state)
            time.sleep(1)

        results = []
        result_set = client.get_query_results(QueryExecutionId=query_execution_id)[
            "ResultSet"
        ]["Rows"]
        columns = [x["VarCharValue"] for x in result_set[0]["Data"]]
        for i in range(1, len(result_set)):
            row = result_set[i]["Data"]
            row_dict = {}
            for col_num in range(len(row)):
                row_dict[columns[col_num]] = row[col_num]["VarCharValue"]
            results.append(row_dict)
        return results

    def _get_columns(
        self, query_result: List[Dict[str, Any]]
    ) -> Tuple[List[str], List[str]]:
        content_columns = []
        metadata_columns = []
        all_columns = list(query_result[0].keys())
        for key in all_columns:
            if key in self.metadata_columns:
                metadata_columns.append(key)
            else:
                content_columns.append(key)

        return content_columns, metadata_columns

    def lazy_load(self) -> Iterator[Document]:
        query_result = self._execute_query()
        content_columns, metadata_columns = self._get_columns(query_result)
        for row in query_result:
            page_content = "\n".join(
                f"{k}: {v}" for k, v in row.items() if k in content_columns
            )
            metadata = {k: v for k, v in row.items() if k in metadata_columns}
            doc = Document(page_content=page_content, metadata=metadata)
            yield doc

    def load(self) -> List[Document]:
        """Load data into document objects."""
        return list(self.lazy_load())
