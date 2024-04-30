from __future__ import annotations

import io
import json
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple

from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader


class AthenaLoader(BaseLoader):
    """Load documents from `AWS Athena`.

    Each document represents one row of the result.
    - By default, all columns are written into the `page_content` of the document
    and none into the `metadata` of the document.
    - If `metadata_columns` are provided then these columns are written
    into the `metadata` of the document while the rest of the columns
    are written into the `page_content` of the document.

    To authenticate, the AWS client uses this method to automatically load credentials:
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
        self.metadata_columns = metadata_columns if metadata_columns is not None else []

        try:
            import boto3
        except ImportError:
            raise ImportError(
                "Could not import boto3 python package. "
                "Please install it with `pip install boto3`."
            )

        try:
            session = (
                boto3.Session(profile_name=profile_name)
                if profile_name is not None
                else boto3.Session()
            )
        except Exception as e:
            raise ValueError(
                "Could not load credentials to authenticate with AWS client. "
                "Please check that credentials in the specified "
                "profile name are valid."
            ) from e

        self.athena_client = session.client("athena")
        self.s3_client = session.client("s3")

    def _execute_query(self) -> List[Dict[str, Any]]:
        response = self.athena_client.start_query_execution(
            QueryString=self.query,
            QueryExecutionContext={"Database": self.database},
            ResultConfiguration={"OutputLocation": self.s3_output_uri},
        )
        query_execution_id = response["QueryExecutionId"]
        while True:
            response = self.athena_client.get_query_execution(
                QueryExecutionId=query_execution_id
            )
            state = response["QueryExecution"]["Status"]["State"]
            if state == "SUCCEEDED":
                break
            elif state == "FAILED":
                resp_status = response["QueryExecution"]["Status"]
                state_change_reason = resp_status["StateChangeReason"]
                err = f"Query Failed: {state_change_reason}"
                raise Exception(err)
            elif state == "CANCELLED":
                raise Exception("Query was cancelled by the user.")
            time.sleep(1)

        result_set = self._get_result_set(query_execution_id)
        return json.loads(result_set.to_json(orient="records"))

    def _remove_suffix(self, input_string: str, suffix: str) -> str:
        if suffix and input_string.endswith(suffix):
            return input_string[: -len(suffix)]
        return input_string

    def _remove_prefix(self, input_string: str, suffix: str) -> str:
        if suffix and input_string.startswith(suffix):
            return input_string[len(suffix) :]
        return input_string

    def _get_result_set(self, query_execution_id: str) -> Any:
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "Could not import pandas python package. "
                "Please install it with `pip install pandas`."
            )

        output_uri = self.s3_output_uri
        tokens = self._remove_prefix(
            self._remove_suffix(output_uri, "/"), "s3://"
        ).split("/")
        bucket = tokens[0]
        key = "/".join(tokens[1:] + [query_execution_id]) + ".csv"

        obj = self.s3_client.get_object(Bucket=bucket, Key=key)
        df = pd.read_csv(io.BytesIO(obj["Body"].read()), encoding="utf8")
        return df

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
            metadata = {
                k: v for k, v in row.items() if k in metadata_columns and v is not None
            }
            doc = Document(page_content=page_content, metadata=metadata)
            yield doc
