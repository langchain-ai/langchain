"""Test WatsonxLLM API wrapper."""

import os

from langchain_ibm import WatsonxLLM

os.environ.pop("WATSONX_APIKEY", None)
os.environ.pop("WATSONX_PROJECT_ID", None)


def test_initialize_watsonxllm_bad_path_without_url() -> None:
    try:
        WatsonxLLM(
            model_id="google/flan-ul2",
        )
    except ValueError as e:
        assert "WATSONX_URL" in e.__str__()


def test_initialize_watsonxllm_cloud_bad_path() -> None:
    try:
        WatsonxLLM(model_id="google/flan-ul2", url="https://us-south.ml.cloud.ibm.com")
    except ValueError as e:
        assert "WATSONX_APIKEY" in e.__str__()


def test_initialize_watsonxllm_cpd_bad_path_without_all() -> None:
    try:
        WatsonxLLM(
            model_id="google/flan-ul2",
            url="https://cpd-zen.apps.cpd48.cp.fyre.ibm.com",
        )
    except ValueError as e:
        assert (
            "WATSONX_APIKEY" in e.__str__()
            and "WATSONX_PASSWORD" in e.__str__()
            and "WATSONX_TOKEN" in e.__str__()
        )


def test_initialize_watsonxllm_cpd_bad_path_password_without_username() -> None:
    try:
        WatsonxLLM(
            model_id="google/flan-ul2",
            url="https://cpd-zen.apps.cpd48.cp.fyre.ibm.com",
            password="test_password",
        )
    except ValueError as e:
        assert "WATSONX_USERNAME" in e.__str__()


def test_initialize_watsonxllm_cpd_bad_path_apikey_without_username() -> None:
    try:
        WatsonxLLM(
            model_id="google/flan-ul2",
            url="https://cpd-zen.apps.cpd48.cp.fyre.ibm.com",
            apikey="test_apikey",
        )
    except ValueError as e:
        assert "WATSONX_USERNAME" in e.__str__()
