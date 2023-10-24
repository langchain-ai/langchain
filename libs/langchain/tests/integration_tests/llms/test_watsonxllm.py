"""Test WatsonxLLM API wrapper."""

from langchain.llms import WatsonxLLM


def test_watsonxllm_call() -> None:
    watsonxllm = WatsonxLLM(
        model_id="google/flan-ul2",
        credentials={"url": "https://us-south.ml.cloud.ibm.com"},
    )
    response = watsonxllm("What color sunflower is?")
    assert isinstance(response, str)


def test_initialize_watsonxllm_cloud_bad_path() -> None:
    try:
        watsonxllm = WatsonxLLM(
            model_id="google/flan-ul2",
            credentials={"url": "https://us-south.ml.cloud.ibm.com"},
        )
    except ValueError as e:
        assert "WATSONX_APIKEY" in e.__str__()


def test_initialize_watsonxllm_cpd_bad_path_without_all() -> None:
    try:
        watsonxllm = WatsonxLLM(
            model_id="google/flan-ul2",
            credentials={"url": "https://cpd-zen.apps.cpd48.cp.fyre.ibm.com"},
        )
    except ValueError as e:
        assert (
            "APIKEY" in e.__str__()
            and "PASSWORD" in e.__str__()
            and "TOKEN" in e.__str__()
        )


def test_initialize_watsonxllm_cpd_bad_path_password_without_username() -> None:
    try:
        watsonxllm = WatsonxLLM(
            model_id="google/flan-ul2",
            credentials={
                "url": "https://cpd-zen.apps.cpd48.cp.fyre.ibm.com",
                "password": "test_password",
            },
        )
    except ValueError as e:
        assert "USERNAME" in e.__str__()


def test_initialize_watsonxllm_cpd_bad_path_apikey_without_username() -> None:
    try:
        watsonxllm = WatsonxLLM(
            model_id="google/flan-ul2",
            credentials={
                "url": "https://cpd-zen.apps.cpd48.cp.fyre.ibm.com",
                "apikey": "test_apikey",
            },
        )
    except ValueError as e:
        assert "USERNAME" in e.__str__()
