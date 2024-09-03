from __future__ import annotations

import os
import tempfile
from typing import Any, Callable, List, Optional

from langchain_community.document_loaders.unstructured import UnstructuredBaseLoader


class UnstructuredApacheOpendalS3FileLoader(UnstructuredBaseLoader):
    def __init__(
        self,
        key: str,
        bucket: str,
        region_name: str,
        *,
        endpoint_url: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        mode: str = "single",
        post_processors: Optional[List[Callable]] = None,
        **unstructured_kwargs: Any,
    ):
        """Initialize with bucket and key name.

        :param bucket: The name of the S3 bucket.
        :param key: The key of the S3 object.

        :param region_name: The name of the region associated with the client.
            A client is associated with a single region.

        :param endpoint_url: The complete URL to use for the constructed
            client.

        :param aws_access_key_id: The access key to use when creating
            the client.  This is entirely optional, and if not provided,
            the credentials configured for the session will automatically
            be used.  You only need to provide this argument if you want
            to override the credentials used for this specific client.

        :param aws_secret_access_key: The secret key to use when creating
            the client.  Same semantics as aws_access_key_id above.

        :param aws_session_token: The session token to use when creating
            the client.  Same semantics as aws_access_key_id above.

        :param mode: Mode in which to read the file. Valid options are: single,
            paged and elements.
        :param post_processors: Post processing functions to be applied to
            extracted elements.
        :param **unstructured_kwargs: Arbitrary additional kwargs to pass in when
            calling `partition`
        """
        super().__init__(mode, post_processors, **unstructured_kwargs)
        self.key = key
        self.bucket = bucket
        self.region_name = region_name
        if endpoint_url is None:
            endpoint_url = ""
        self.endpoint_url = endpoint_url
        if aws_access_key_id is None:
            aws_access_key_id = ""
        self.aws_access_key_id = aws_access_key_id
        if aws_secret_access_key is None:
            aws_secret_access_key = ""
        self.aws_secret_access_key = aws_secret_access_key
        if aws_session_token is None:
            aws_session_token = ""
        self.aws_session_token = aws_session_token

    def _get_metadata(self) -> dict:
        return {"source": f"s3://{self.bucket}/{self.key}"}

    def _get_elements(self) -> List:
        """Get elements."""
        from unstructured.partition.auto import partition

        try:
            from opendal import Operator
        except ImportError:
            raise ImportError(
                "Could not import `opendal` python package. "
                "Please install it with `pip install opendal`."
            )
        op = Operator(
            "s3",
            endpoint=self.endpoint_url,
            bucket=self.bucket,
            region=self.region_name,
            access_key_id=self.aws_access_key_id,
            secret_access_key=self.aws_secret_access_key,
            session_token=self.aws_session_token,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = f"{temp_dir}/{self.key}"
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            file_buffer = op.read(path=self.key)
            with open(file_path, mode="wb") as file:
                file.write(bytes(file_buffer))

            return partition(filename=file_path, **self.unstructured_kwargs)
