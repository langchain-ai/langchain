"""
Test Amazon Bedrock API wrapper and services i.e 'Guardrails for Amazon Bedrock'.
You can get a list of models from the bedrock client by running 'bedrock_models()'

"""

import os
from typing import Any

import pytest
from langchain_core.callbacks import AsyncCallbackHandler

from langchain_community.llms.bedrock import Bedrock

# this is the guardrails id for the model you want to test
GUARDRAILS_ID = os.environ.get("GUARDRAILS_ID", "7jarelix77")
# this is the guardrails version for the model you want to test
GUARDRAILS_VERSION = os.environ.get("GUARDRAILS_VERSION", "1")
# this should trigger the guardrails - you can change this to any text you want which
# will trigger the guardrails
GUARDRAILS_TRIGGER = os.environ.get(
    "GUARDRAILS_TRIGGERING_QUERY", "I want to talk about politics."
)


class BedrockAsyncCallbackHandler(AsyncCallbackHandler):
    """Async callback handler that can be used to handle callbacks from langchain."""

    guardrails_intervened: bool = False

    async def on_llm_error(
        self,
        error: BaseException,
        **kwargs: Any,
    ) -> Any:
        reason = kwargs.get("reason")
        if reason == "GUARDRAIL_INTERVENED":
            self.guardrails_intervened = True

    def get_response(self):  # type: ignore[no-untyped-def]
        return self.guardrails_intervened


@pytest.fixture(autouse=True)
def bedrock_runtime_client():  # type: ignore[no-untyped-def]
    import boto3

    try:
        client = boto3.client(
            "bedrock-runtime",
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
        )
        return client
    except Exception as e:
        pytest.fail(f"can not connect to bedrock-runtime client: {e}", pytrace=False)


@pytest.fixture(autouse=True)
def bedrock_client():  # type: ignore[no-untyped-def]
    import boto3

    try:
        client = boto3.client(
            "bedrock",
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
        )
        return client
    except Exception as e:
        pytest.fail(f"can not connect to bedrock client: {e}", pytrace=False)


@pytest.fixture
def bedrock_models(bedrock_client):  # type: ignore[no-untyped-def]
    """List bedrock models."""
    response = bedrock_client.list_foundation_models().get("modelSummaries")
    models = {}
    for model in response:
        models[model.get("modelId")] = model.get("modelName")
    return models


def test_claude_instant_v1(bedrock_runtime_client, bedrock_models):  # type: ignore[no-untyped-def]
    try:
        llm = Bedrock(
            model_id="anthropic.claude-instant-v1",
            client=bedrock_runtime_client,
            model_kwargs={},
        )
        output = llm.invoke("Say something positive:")
        assert isinstance(output, str)
    except Exception as e:
        pytest.fail(f"can not instantiate claude-instant-v1: {e}", pytrace=False)


def test_amazon_bedrock_guardrails_no_intervention_for_valid_query(  # type: ignore[no-untyped-def]
    bedrock_runtime_client, bedrock_models
):
    try:
        llm = Bedrock(
            model_id="anthropic.claude-instant-v1",
            client=bedrock_runtime_client,
            model_kwargs={},
            guardrails={
                "id": GUARDRAILS_ID,
                "version": GUARDRAILS_VERSION,
                "trace": False,
            },
        )
        output = llm.invoke("Say something positive:")
        assert isinstance(output, str)
    except Exception as e:
        pytest.fail(f"can not instantiate claude-instant-v1: {e}", pytrace=False)


def test_amazon_bedrock_guardrails_intervention_for_invalid_query(  # type: ignore[no-untyped-def]
    bedrock_runtime_client, bedrock_models
):
    try:
        handler = BedrockAsyncCallbackHandler()
        llm = Bedrock(
            model_id="anthropic.claude-instant-v1",
            client=bedrock_runtime_client,
            model_kwargs={},
            guardrails={
                "id": GUARDRAILS_ID,
                "version": GUARDRAILS_VERSION,
                "trace": True,
            },
            callbacks=[handler],
        )
    except Exception as e:
        pytest.fail(f"can not instantiate claude-instant-v1: {e}", pytrace=False)
    else:
        llm.invoke(GUARDRAILS_TRIGGER)
        guardrails_intervened = handler.get_response()
        assert guardrails_intervened is True
