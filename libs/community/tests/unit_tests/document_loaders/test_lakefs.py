import unittest
from typing import Any
from unittest.mock import patch

import pytest
import requests_mock
from requests_mock.mocker import Mocker

from langchain_community.document_loaders.lakefs import LakeFSLoader


@pytest.fixture
def mock_lakefs_client() -> Any:
    with patch(
        "langchain_community.document_loaders.lakefs.LakeFSClient"
    ) as mock_lakefs_client:
        mock_lakefs_client.return_value.ls_objects.return_value = [
            ("path_bla.txt", "https://physical_address_bla")
        ]
        mock_lakefs_client.return_value.is_presign_supported.return_value = True
        yield mock_lakefs_client.return_value


@pytest.fixture
def mock_lakefs_client_no_presign_not_local() -> Any:
    with patch(
        "langchain_community.document_loaders.lakefs.LakeFSClient"
    ) as mock_lakefs_client:
        mock_lakefs_client.return_value.ls_objects.return_value = [
            ("path_bla.txt", "https://physical_address_bla")
        ]
        mock_lakefs_client.return_value.is_presign_supported.return_value = False
        yield mock_lakefs_client.return_value


@pytest.fixture
def mock_unstructured_local() -> Any:
    with patch(
        "langchain_community.document_loaders.lakefs.UnstructuredLakeFSLoader"
    ) as mock_unstructured_lakefs:
        mock_unstructured_lakefs.return_value.load.return_value = [
            ("text content", "pdf content")
        ]
        yield mock_unstructured_lakefs.return_value


@pytest.fixture
def mock_lakefs_client_no_presign_local() -> Any:
    with patch(
        "langchain_community.document_loaders.lakefs.LakeFSClient"
    ) as mock_lakefs_client:
        mock_lakefs_client.return_value.ls_objects.return_value = [
            ("path_bla.txt", "local:///physical_address_bla")
        ]
        mock_lakefs_client.return_value.is_presign_supported.return_value = False
        yield mock_lakefs_client.return_value


class TestLakeFSLoader(unittest.TestCase):
    lakefs_access_key: str = "lakefs_access_key"
    lakefs_secret_key: str = "lakefs_secret_key"
    endpoint: str = "endpoint"
    repo: str = "repo"
    ref: str = "ref"
    path: str = "path"

    @pytest.mark.usefixtures("mock_lakefs_client_no_presign_not_local")
    def test_non_presigned_loading_fail(self) -> None:
        loader = LakeFSLoader(
            self.lakefs_access_key, self.lakefs_secret_key, self.endpoint
        )
        loader.set_repo(self.repo)
        loader.set_ref(self.ref)
        loader.set_path(self.path)
        with pytest.raises(ImportError):
            loader.load()

    @pytest.mark.usefixtures(
        "mock_lakefs_client_no_presign_local", "mock_unstructured_local"
    )
    def test_non_presigned_loading(self) -> None:
        loader = LakeFSLoader(
            lakefs_access_key="lakefs_access_key",
            lakefs_secret_key="lakefs_secret_key",
            lakefs_endpoint=self.endpoint,
        )
        loader.set_repo(self.repo)
        loader.set_ref(self.ref)
        loader.set_path(self.path)
        loader.load()
