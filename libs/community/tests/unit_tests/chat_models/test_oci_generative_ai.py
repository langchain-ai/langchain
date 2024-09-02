"""Test OCI Generative AI LLM service"""

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import HumanMessage
from pytest import MonkeyPatch

from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI


class MockResponseDict(dict):
    def __getattr__(self, val):  # type: ignore[no-untyped-def]
        return self[val]


@pytest.mark.requires("oci")
@pytest.mark.parametrize(
    "test_model_id", ["cohere.command-r-16k", "meta.llama-3-70b-instruct"]
)
def test_llm_chat(monkeypatch: MonkeyPatch, test_model_id: str) -> None:
    """Test valid chat call to OCI Generative AI LLM service."""
    oci_gen_ai_client = MagicMock()
    llm = ChatOCIGenAI(model_id=test_model_id, client=oci_gen_ai_client)

    provider = llm.model_id.split(".")[0].lower()

    def mocked_response(*args):  # type: ignore[no-untyped-def]
        response_text = "Assistant chat reply."
        response = None
        if provider == "cohere":
            response = MockResponseDict(
                {
                    "status": 200,
                    "data": MockResponseDict(
                        {
                            "chat_response": MockResponseDict(
                                {
                                    "text": response_text,
                                    "finish_reason": "completed",
                                    "is_search_required": None,
                                    "search_queries": None,
                                    "citations": None,
                                    "documents": None,
                                    "tool_calls": None,
                                }
                            ),
                            "model_id": "cohere.command-r-16k",
                            "model_version": "1.0.0",
                        }
                    ),
                    "request_id": "1234567890",
                    "headers": MockResponseDict(
                        {
                            "content-length": "123",
                        }
                    ),
                }
            )
        elif provider == "meta":
            response = MockResponseDict(
                {
                    "status": 200,
                    "data": MockResponseDict(
                        {
                            "chat_response": MockResponseDict(
                                {
                                    "choices": [
                                        MockResponseDict(
                                            {
                                                "message": MockResponseDict(
                                                    {
                                                        "content": [
                                                            MockResponseDict(
                                                                {
                                                                    "text": response_text,  # noqa: E501
                                                                }
                                                            )
                                                        ]
                                                    }
                                                ),
                                                "finish_reason": "completed",
                                            }
                                        )
                                    ],
                                    "time_created": "2024-09-01T00:00:00Z",
                                }
                            ),
                            "model_id": "cohere.command-r-16k",
                            "model_version": "1.0.0",
                        }
                    ),
                    "request_id": "1234567890",
                    "headers": MockResponseDict(
                        {
                            "content-length": "123",
                        }
                    ),
                }
            )
        return response

    monkeypatch.setattr(llm.client, "chat", mocked_response)

    messages = [
        HumanMessage(content="User message"),
    ]

    expected = "Assistant chat reply."
    actual = llm.invoke(messages, temperature=0.2)
    assert actual.content == expected
