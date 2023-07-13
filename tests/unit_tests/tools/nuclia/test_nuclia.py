import base64
import json
import os
from unittest import mock

from langchain.tools.nuclia.tool import NucliaUnderstandingAPI

class FakeUploadResponse:
    status_code = 200
    text = "fake_uuid"

class FakePushResponse:
    status_code = 200
    def json(self):
        return {"uuid": "fake_uuid"}

class FakePullResponse:
    status_code = 200
    def json(self):
        return {"status": "ok", "payload": base64.b64encode(bytes("{\"some\": \"data\"}}", "utf-8"))}

def FakeParseFromString(**args):
    def ParseFromString(self, data):
        self.uuid = "fake_uuid"
    return ParseFromString

class FakeRequest:
    def post(self, url, **kwargs):
        if url.endswith("/processing/upload"):
            return FakeUploadResponse()
        elif url.endswith("/processing/push"):
            return FakePushResponse()
        else:
            raise Exception("Invalid URL")

    def get(self, url, **kwargs):
        if url.endswith("/processing/pull"):
            return FakePullResponse()
        else:
            raise Exception("Invalid URL")

@mock.patch.dict(os.environ, {"NUCLIA_NUA_KEY": "_a_key_"})
def test_nuclia_tool() -> None:
    with mock.patch('nucliadb_protos.writer_pb2.BrokerMessage.ParseFromString', new_callable=FakeParseFromString):
        nua = NucliaUnderstandingAPI()
        nua.requests = FakeRequest()
        uuid = nua.run({"action": "push", "id": "1", "path": "/Users/ebr/dev/nuclia/docs/README.md", "enable_ml": False})
        assert uuid == "fake_uuid"
        data = nua.run({"action": "pull", "id": "1", "path": None, "enable_ml": False})
        assert json.loads(data)["uuid"] == "fake_uuid"
