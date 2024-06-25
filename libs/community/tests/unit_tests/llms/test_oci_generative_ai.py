"""Test OCI Generative AI LLM service"""
from unittest.mock import MagicMock

import pytest
from pytest import MonkeyPatch

from langchain_community.llms.oci_generative_ai import OCIGenAI


class MockResponseDict(dict):
    def __getattr__(self, val):  # type: ignore[no-untyped-def]
        return self[val]


@pytest.mark.requires("oci")
@pytest.mark.parametrize(
    "test_model_id", ["cohere.command", "cohere.command-light", "meta.llama-2-70b-chat"]
)
def test_llm_complete(monkeypatch: MonkeyPatch, test_model_id: str) -> None:
    """Test valid completion call to OCI Generative AI LLM service."""
    oci_gen_ai_client = MagicMock()
    llm = OCIGenAI(model_id=test_model_id, client=oci_gen_ai_client)

    provider = llm.model_id.split(".")[0].lower()

    def mocked_response(*args):  # type: ignore[no-untyped-def]
        response_text = "This is the completion."

        if provider == "cohere":
            return MockResponseDict(
                {
                    "status": 200,
                    "data": MockResponseDict(
                        {
                            "inference_response": MockResponseDict(
                                {
                                    "generated_texts": [
                                        MockResponseDict(
                                            {
                                                "text": response_text,
                                            }
                                        )
                                    ]
                                }
                            )
                        }
                    ),
                }
            )

        if provider == "meta":
            return MockResponseDict(
                {
                    "status": 200,
                    "data": MockResponseDict(
                        {
                            "inference_response": MockResponseDict(
                                {
                                    "choices": [
                                        MockResponseDict(
                                            {
                                                "text": response_text,
                                            }
                                        )
                                    ]
                                }
                            )
                        }
                    ),
                }
            )

    monkeypatch.setattr(llm.client, "generate_text", mocked_response)
    output = llm.invoke("This is a prompt.", temperature=0.2)
    assert output == "This is the completion."
