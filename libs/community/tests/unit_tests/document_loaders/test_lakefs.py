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
    endpoint: str = "http://localhost:8000"
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
        with pytest.raises(ValueError):
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

    @requests_mock.Mocker()
    @pytest.mark.requires("unstructured")
    def test_load_data(self, mocker: Mocker) -> None:
        mocker.register_uri(requests_mock.ANY, requests_mock.ANY, status_code=200)
        mocker.register_uri(
            "GET", f"{self.endpoint}/api/v1/healthcheck", status_code=200
        )
        mock_results = [
            {
                "path": "books/sample1.txt",
                "physical_address": "local://fake/path/sample1.txt",
            },
            {
                "path": "books/sample2.txt",
                "physical_address": "local://fake/path/sample2.txt",
            },
        ]

        mock_response = {
            "pagination": {
                "has_more": False,
                "max_per_page": 1000,
                "next_offset": "",
                "results": len(mock_results),
            },
            "results": mock_results,
        }
        mock_config_response = {
            "storage_config": {
                "pre_sign_support": True  # or False, depending on your test case
            }
        }
        mocker.register_uri(
            "GET",
            f"{self.endpoint}/api/v1/repositories/{self.repo}/refs/{self.ref}/objects/ls?",
            json=mock_response,
        )
        mocker.get(f"{self.endpoint}/api/v1/config", json=mock_config_response)
        loader = LakeFSLoader(
            lakefs_access_key="lakefs_access_key",
            lakefs_secret_key="lakefs_secret_key",
            lakefs_endpoint=self.endpoint,
        )

        loader.set_repo(self.repo)
        loader.set_ref(self.ref)
        loader.set_path(self.path)
        documents = loader.load()
        self.assertEqual(len(documents), 2)
