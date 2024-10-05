import uuid

import pytest

from langchain_core.documents.base import Document
from langchain_community.callbacks.tracers.wandb import _serialize_io
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompt_values import StringPromptValue


# installed by `pip install protobuf`
@pytest.mark.requires("google.protobuf")
def test_serialization_google_protobuf_message():
    from google.protobuf.json_format import MessageToJson
    from tests.unit_tests.callbacks.tracers._human_message_protobuf_pb2 import (
        HumanMessage as ProtobufHumanMessage,
    )

    pb_human_message = ProtobufHumanMessage()
    pb_human_message.content = "Hello, world!"
    pb_human_message.type = "human"
    pb_human_message.id = str(uuid.uuid4())

    run_io = {'input': pb_human_message}
    serialized = _serialize_io(run_io)
    expected = {'input': MessageToJson(pb_human_message)}
    assert expected == serialized


def test_serialization_prompt():
    prompt_value = StringPromptValue(text="Hello, world!")
    run_io = {'input': prompt_value}
    serialized = _serialize_io(run_io)
    expected = {'input': prompt_value.json()}
    assert expected == serialized


def test_serialization_message():
    message = HumanMessage(content="Hello, world!")
    run_io = {'input': message}
    serialized = _serialize_io(run_io)
    expected = {'input': message.json()}
    assert expected == serialized


def test_serialization_list_of_messages():
    messages = [
        SystemMessage(content="You are a helpful AI assistant"),
        HumanMessage(content="Hello, world!")
    ]
    run_io = {'input': messages}
    serialized = _serialize_io(run_io)
    expected = {'input': [m.json() for m in messages]}
    assert expected == serialized


def test_serialization_documents():
    documents = [
        Document(page_content="Hello, world!", metadata={"id": "doc0"}),
        Document(page_content="Let's go!", metadata={"id": "doc1"})
    ]
    run_io = {'input_documents': documents}
    serialized = _serialize_io(run_io)
    expected = {
        'input_document_0': documents[0].json(),
        'input_document_1': documents[1].json(),
    }
    assert expected == serialized
