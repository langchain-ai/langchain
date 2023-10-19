from typing import Dict

import pytest
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


def mocked_requests_post(url: str, headers: dict, json: dict) -> MockResponse:
    assert url.startswith(_GRADIENT_BASE_URL)
    assert _MODEL_ID in url
    assert json
    assert headers

    assert headers.get("authorization") == f"Bearer {_GRADIENT_SECRET}"
    assert headers.get("x-gradient-workspace-id") == f"{_GRADIENT_WORKSPACE_ID}"
    query = json.get("query")
    assert query and isinstance(query, str)
    output = "bar" if "foo" in query else "baz"

    return MockResponse(
        json_data={"generatedOutput": output},
        status_code=200,
    )


@pytest.mark.parametrize(
    "setup",
    [
        dict(
            gradient_api_url=_GRADIENT_BASE_URL,
            gradient_access_token=_GRADIENT_SECRET,
            gradient_workspace_id=_GRADIENT_WORKSPACE_ID,
            model=_MODEL_ID,
        ),
        dict(
            gradient_api_url=_GRADIENT_BASE_URL,
            gradient_access_token=_GRADIENT_SECRET,
            gradient_workspace_id=_GRADIENT_WORKSPACE_ID,
            model_id=_MODEL_ID,
        ),
    ],
)
def test_gradient_llm_sync(mocker: MockerFixture, setup: dict) -> None:
    mocker.patch("requests.post", side_effect=mocked_requests_post)

    llm = GradientLLM(**setup)
    assert llm.gradient_access_token == _GRADIENT_SECRET
    assert llm.gradient_api_url == _GRADIENT_BASE_URL
    assert llm.gradient_workspace_id == _GRADIENT_WORKSPACE_ID
    assert llm.model_id == _MODEL_ID

    response = llm("Say foo:")
    want = "bar"

    assert response == want


@pytest.mark.parametrize(
    "setup",
    [
        dict(
            gradient_api_url=_GRADIENT_BASE_URL,
            gradient_access_token=_GRADIENT_SECRET,
            gradient_workspace_id=_GRADIENT_WORKSPACE_ID,
            model=_MODEL_ID,
        )
    ],
)
def test_gradient_llm_sync_batch(mocker: MockerFixture, setup: dict) -> None:
    mocker.patch("requests.post", side_effect=mocked_requests_post)

    llm = GradientLLM(**setup)
    assert llm.gradient_access_token == _GRADIENT_SECRET
    assert llm.gradient_api_url == _GRADIENT_BASE_URL
    assert llm.gradient_workspace_id == _GRADIENT_WORKSPACE_ID
    assert llm.model_id == _MODEL_ID

    inputs = ["Say foo:", "Say baz:", "Say foo again"]
    response = llm._generate(inputs)

    want = ["bar", "baz", "bar"]
    assert len(response.generations) == len(inputs)
    for i, gen in enumerate(response.generations):
        assert gen[0].text == want[i]
