from typing import Dict

from pytest_mock import MockerFixture

from langchain.llms import GradientLLM

_MODEL_ID = "my_model_valid_id"
_GRADIENT_SECRET = "secret_valid_token_123456"
_GRADIENT_WORKSPACE_ID = "valid_workspace_12345"
_GRADIENT_BASE_URL = "https://api.gradient.ai/api"


class MockResponse:
    def __init__(self, json_data: Dict, status_code: int):
        self.json_data = json_data
        self.status_code = status_code

    def json(self) -> Dict:
        return self.json_data


def mocked_requests_post(
    url: str,
    headers: dict,
    json: dict,
) -> MockResponse:
    assert url.startswith(_GRADIENT_BASE_URL)
    assert headers
    assert json

    return MockResponse(
        json_data={"generatedOutput": "bar"},
        status_code=200,
    )


def test_gradient_llm_sync(
    mocker: MockerFixture,
) -> None:
    mocker.patch("requests.post", side_effect=mocked_requests_post)

    llm = GradientLLM(
        gradient_api_url=_GRADIENT_BASE_URL,
        gradient_access_token=_GRADIENT_SECRET,
        gradient_workspace_id=_GRADIENT_WORKSPACE_ID,
        model_id=_MODEL_ID,
    )
    assert llm.gradient_access_token == _GRADIENT_SECRET
    assert llm.gradient_api_url == _GRADIENT_BASE_URL
    assert llm.gradient_workspace_id == _GRADIENT_WORKSPACE_ID
    assert llm.model_id == _MODEL_ID

    response = llm("Say foo:")
    want = "bar"

    assert response == want
