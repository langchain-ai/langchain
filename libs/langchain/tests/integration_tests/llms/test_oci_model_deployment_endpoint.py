"""Test OCIModelDeploymentVLLM Endpoint."""

import pytest
from tests.integration_tests.llms.utils import assert_llm_equality

from langchain.llms.oci_data_science_model_deployment_endpoint import (
    OCIModelDeploymentTGI,
    OCIModelDeploymentVLLM,
)
from langchain.load import dumpd, dumps
from langchain.load.load import load, loads


@pytest.mark.skip(
    "This test requires an inference endpoint. Tested with OCI Data Science Model Deployment endpoints."
)
def test_call_vllm() -> None:
    """Test valid call to oci model deployment endpoint."""
    llm = OCIModelDeploymentVLLM(endpoint="")
    output = llm("Who is the first president of United States?")
    print(output)
    assert isinstance(output, str)


@pytest.mark.skip(
    "This test requires an inference endpoint. Tested with OCI Data Science Model Deployment endpoints."
)
def test_call_tgi() -> None:
    """Test valid call to oci model deployment endpoint."""
    llm = OCIModelDeploymentTGI(endpoint="")
    output = llm("Who is the first president of United States?")
    print(output)
    assert isinstance(output, str)


@pytest.mark.requires("oracle-ads")
def test_dumpd_load_tgi() -> None:
    """Test serialization/deserialization (dumpd/load) an OCIModelDeploymentTGI LLM."""
    llm = OCIModelDeploymentTGI(
        endpoint="https://<MD_OCID>/predict",
        temperature=0.75,
        max_tokens=100,
        k=1,
    )
    loaded_llm = load(dumpd(llm))
    assert_llm_equality(llm, loaded_llm, exclude=["auth"])


@pytest.mark.requires("oracle-ads")
def test_dumps_loads_vllm() -> None:
    """Test serialization/deserialization (dumps/loads) an OCIModelDeploymentVLLM LLM."""
    llm = OCIModelDeploymentVLLM(
        endpoint="https://<MD_OCID>/predict",
        model="mymodel",
        n=2,
        temperature=0.75,
        max_tokens=100,
        k=1,
    )
    loaded_llm = loads(dumps(llm))
    assert_llm_equality(llm, loaded_llm, exclude=["auth"])
