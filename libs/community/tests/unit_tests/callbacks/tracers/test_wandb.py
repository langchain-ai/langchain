import uuid

from google.protobuf.json_format import MessageToJson

from documents import Document
from langchain_community.callbacks.tracers.wandb import _serialize_io
from langchain_core.messages import HumanMessage
from messages import SystemMessage
from tests.unit_tests.callbacks.tracers._human_message_protobuf_pb2 import (
    HumanMessage as ProtobufHumanMessage,
)


def test_input_google_protobuf_message():
    pb_human_message = ProtobufHumanMessage()
    pb_human_message.content = "Hello, world!"
    pb_human_message.type = "human"
    pb_human_message.id = str(uuid.uuid4())

    run_io = {'input': pb_human_message}
    serialized = _serialize_io(run_io)
    expected = {'input': MessageToJson(pb_human_message)}
    assert expected == serialized


def test_input_message():
    message = HumanMessage(content="Hello, world!")
    run_io = {'input': message}
    serialized = _serialize_io(run_io)
    expected = {'input': message.json()}
    assert expected == serialized


def test_input_list_of_messages():
    messages = [
        SystemMessage(content="You are a helpful AI assistant"),
        HumanMessage(content="Hello, world!")
    ]
    run_io = {'input': messages}
    serialized = _serialize_io(run_io)
    expected = {'input': [m.json() for m in messages]}
    assert expected == serialized


def test_input_documents():
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
