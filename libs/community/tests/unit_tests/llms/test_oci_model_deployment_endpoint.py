"""Test OCI Data Science Model Deployment Endpoint."""
import pytest
import responses

from langchain_community.llms import OCIModelDeploymentTGI, OCIModelDeploymentVLLM


@pytest.mark.requires("oracle_ads")
@responses.activate
def test_call_vllm() -> None:
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

    llm = OCIModelDeploymentVLLM(endpoint=endpoint, model="my_model")
    output = llm.invoke("This is a prompt.")
    assert isinstance(output, str)


@pytest.mark.requires("oracle_ads")
@responses.activate
def test_call_tgi() -> None:
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

    llm = OCIModelDeploymentTGI(endpoint=endpoint)
    output = llm.invoke("This is a prompt.")
    assert isinstance(output, str)
