import pytest
from opensearchpy import RequestsAWSV4SignerAuth
from pytest_mock import MockerFixture
from requests_aws4auth import AWS4Auth

from langchain_community.vectorstores.opensearch_vector_search import OpenSearchVectorSearch
from langchain_community.embeddings import FakeEmbeddings


@pytest.mark.requires("opensearchpy")
@pytest.mark.parametrize(["service", "expected"], [("aoss", True), ("es", False)])
def test_detect_aoss_using_signer_auth(
    mocker: MockerFixture, service: str, expected: bool
) -> None:
    mocker.patch.object(RequestsAWSV4SignerAuth, "_sign_request")
    http_auth = RequestsAWSV4SignerAuth(
        credentials="credentials", region="eu-central-1", service=service
    )
    database = OpenSearchVectorSearch(
        opensearch_url="http://localhost:9200",
        index_name="test",
        embedding_function=FakeEmbeddings(size=42),
        http_auth=http_auth,
    )

    assert database.is_aoss == expected


@pytest.mark.requires("opensearchpy")
@pytest.mark.requires("requests_aws4auth")
@pytest.mark.parametrize(["service", "expected"], [("aoss", True), ("es", False)])
def test_detect_aoss_using_aws4auth(service: str, expected: bool) -> None:
    http_auth = AWS4Auth("access_key_id", "secret_access_key", "eu-central-1", service)
    database = OpenSearchVectorSearch(
        opensearch_url="http://localhost:9200",
        index_name="test",
        embedding_function=FakeEmbeddings(size=42),
        http_auth=http_auth,
    )

    assert database.is_aoss == expected


@pytest.mark.requires("opensearchpy")
def test_detect_aoss_using_no_auth() -> None:
    database = OpenSearchVectorSearch(
        opensearch_url="http://localhost:9200",
        index_name="test",
        embedding_function=FakeEmbeddings(size=42),
    )

    assert database.is_aoss is False
