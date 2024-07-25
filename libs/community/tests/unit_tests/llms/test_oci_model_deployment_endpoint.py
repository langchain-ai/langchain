"""Test OCI Data Science Model Deployment Endpoint."""

import pytest
import responses
from pytest_mock import MockerFixture

from langchain_community.llms import OCIModelDeploymentTGI, OCIModelDeploymentVLLM


@pytest.mark.requires("ads")
@responses.activate
def test_call_vllm(mocker: MockerFixture) -> None:
    """Test valid call to oci model deployment endpoint."""
    endpoint = "https://MD_OCID/predict"
    responses.add(
        responses.POST,
        endpoint,
        json={
            "choices": [{"index": 0, "text": "This is a completion."}],
        },
        status=200,
    )
    mocker.patch("ads.common.auth.default_signer", return_value=dict(signer=None))

    llm = OCIModelDeploymentVLLM(endpoint=endpoint, model="my_model")
    output = llm.invoke("This is a prompt.")
    assert isinstance(output, str)


@pytest.mark.requires("ads")
@responses.activate
def test_call_tgi(mocker: MockerFixture) -> None:
    """Test valid call to oci model deployment endpoint."""
    endpoint = "https://MD_OCID/predict"
    responses.add(
        responses.POST,
        endpoint,
        json={
            "generated_text": "This is a completion.",
        },
        status=200,
    )
    mocker.patch("ads.common.auth.default_signer", return_value=dict(signer=None))

    llm = OCIModelDeploymentTGI(endpoint=endpoint)
    output = llm.invoke("This is a prompt.")
    assert isinstance(output, str)
