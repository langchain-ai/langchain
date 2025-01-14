"""Test OCI Data Science Model Deployment Endpoint."""

import responses
import pytest
from pytest_mock import MockerFixture
from langchain_community.embeddings import OCIModelDeploymentEndpointEmbeddings


@pytest.mark.requires("ads")
@responses.activate
def test_embedding_call(mocker: MockerFixture) -> None:
    """Test valid call to oci model deployment endpoint."""
    endpoint = "https://MD_OCID/predict"
    documents = ["Hello", "World"]
    expected_output = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    responses.add(
        responses.POST,
        endpoint,
        json={
            "embeddings": expected_output,
        },
        status=200,
    )
    mocker.patch("ads.common.auth.default_signer", return_value=dict(signer=None))

    embeddings = OCIModelDeploymentEndpointEmbeddings(  # type: ignore[call-arg]
        endpoint=endpoint,
    )

    output = embeddings.embed_documents(documents)
    assert output == expected_output
