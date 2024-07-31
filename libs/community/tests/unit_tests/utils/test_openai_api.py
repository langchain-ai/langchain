# from datetime import datetime
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from langchain_community.utils.openai_api import OpenAIChatCompletionAPI, Runnable


# Mock the Runnable class
class MockRunnable(Runnable):
    def __init__(self):
        self.model_name = "test_model"

    def invoke(self, input):
        return MagicMock(
            to_json=lambda: {
                "kwargs": {
                    "id": "test_id",
                    "content": "test_content",
                    "response_metadata": {
                        "usage_metadata": {
                            "prompt_token_count": 5,
                            "candidates_token_count": 10,
                            "total_token_count": 15,
                        }
                    },
                }
            }
        )


# Initialize the API with the mock runnable
mock_runnable = MockRunnable()
api = OpenAIChatCompletionAPI(langchain_runnable=mock_runnable)
client = TestClient(api.app)


def test_liveness_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_completion():
    request_data = {
        "model": "test_model",
        "user": "test_user",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    response = client.post("/chat/completions", json=request_data)
    assert response.status_code == 200
    response_data = response.json()
    assert response_data["id"] == "test_id"
    assert response_data["object"] == "chat.completion"
    assert response_data["model"] == "test_model"
    assert response_data["choices"][0]["message"]["content"] == "test_content"
    assert response_data["usage"]["prompt_tokens"] == 5
    assert response_data["usage"]["completion_tokens"] == 10
    assert response_data["usage"]["total_tokens"] == 15
