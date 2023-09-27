import base64
import json
import os
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from langchain.tools.nuclia.tool import NucliaUnderstandingAPI

README_PATH = Path(__file__).parents[4] / "README.md"


class FakeUploadResponse:
    status_code = 200
    text = "fake_uuid"


class FakePushResponse:
    status_code = 200

    def json(self) -> Any:
        return {"uuid": "fake_uuid"}


class FakePullResponse:
    status_code = 200

    def json(self) -> Any:
        return {
            "status": "ok",
            "payload": base64.b64encode(bytes('{"some": "data"}}', "utf-8")),
        }


def FakeParseFromString(**args: Any) -> Any:
    def ParseFromString(self: Any, data: str) -> None:
        self.uuid = "fake_uuid"

    return ParseFromString


def fakepost(**kwargs: Any) -> Any:
    def fn(url: str, **kwargs: Any) -> Any:
        if url.endswith("/processing/upload"):
            return FakeUploadResponse()
        elif url.endswith("/processing/push"):
            return FakePushResponse()
        else:
            raise Exception("Invalid POST URL")

    return fn


def fakeget(**kwargs: Any) -> Any:
    def fn(url: str, **kwargs: Any) -> Any:
        if url.endswith("/processing/pull"):
            return FakePullResponse()
        else:
            raise Exception("Invalid GET URL")

    return fn


@mock.patch.dict(os.environ, {"NUCLIA_NUA_KEY": "_a_key_"})
@pytest.mark.requires("nucliadb_protos")
def test_nuclia_tool() -> None:
    with mock.patch(
        "nucliadb_protos.writer_pb2.BrokerMessage.ParseFromString",
        new_callable=FakeParseFromString,
    ):
        with mock.patch("requests.post", new_callable=fakepost):
            with mock.patch("requests.get", new_callable=fakeget):
                nua = NucliaUnderstandingAPI(enable_ml=False)
                uuid = nua.run(
                    {
                        "action": "push",
                        "id": "1",
                        "path": str(README_PATH),
                        "text": None,
                    }
                )
                assert uuid == "fake_uuid"
                data = nua.run(
                    {"action": "pull", "id": "1", "path": None, "text": None}
                )
                assert json.loads(data)["uuid"] == "fake_uuid"


@pytest.mark.asyncio
@pytest.mark.requires("nucliadb_protos")
async def test_async_call() -> None:
    with mock.patch(
        "nucliadb_protos.writer_pb2.BrokerMessage.ParseFromString",
        new_callable=FakeParseFromString,
    ):
        with mock.patch("requests.post", new_callable=fakepost):
            with mock.patch("requests.get", new_callable=fakeget):
                with mock.patch("os.environ.get", return_value="_a_key_"):
                    nua = NucliaUnderstandingAPI(enable_ml=False)
                    data = await nua.arun(
                        {
                            "action": "push",
                            "id": "1",
                            "path": str(README_PATH),
                            "text": None,
                        }
                    )
                    assert json.loads(data)["uuid"] == "fake_uuid"
