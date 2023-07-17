"""Tool for the Nuclia Understanding API.

Installation:

```bash
    pip install --upgrade protobuf
    pip install nucliadb-protos
```
"""

import base64
import mimetypes
import os
from typing import Any, Dict, Optional, Type, Union

import requests
from google.protobuf.json_format import MessageToJson
from pydantic import BaseModel, Field

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.tools.base import BaseTool

try:
    from nucliadb_protos.writer_pb2 import BrokerMessage
except ImportError:
    raise ImportError(
        "nucliadb-protos is not installed. "
        "Run `pip install nucliadb-protos` to install."
    )


class NUASchema(BaseModel):
    action: str = Field(
        ...,
        description="Action to perform. Either `push` or `pull`.",
    )
    id: str = Field(
        ...,
        description="ID of the file to push or pull.",
    )
    enable_ml: bool = Field(
        ...,
        description="Enable Machine Learning processing "
        "(applicable only to `push` action).",
    )
    path: Optional[str] = Field(
        ...,
        description="Path to the file to push (needed only for `push` action).",
    )


class NucliaUnderstandingAPI(BaseTool):
    """Tool to process files with the Nuclia Understanding API."""

    name = "nuclia_understanding_api"
    description = (
        "A wrapper around Nuclia Understanding API endpoints. "
        "Useful for when you need to extract text from any kind of files. "
    )
    args_schema: Type[BaseModel] = NUASchema
    _results: Dict[str, Any] = {}
    _config: Dict[str, str] = {}

    def __init__(self) -> None:
        zone = os.environ.get("NUCLIA_ZONE", "europe-1")
        self._config["BACKEND"] = f"https://{zone}.nuclia.cloud/api/v1"
        key = os.environ.get("NUCLIA_NUA_KEY")
        if not key:
            raise ValueError("NUCLIA_NUA_KEY environment variable not set")
        else:
            self._config["NUA_KEY"] = key
        super().__init__()

    def _run(
        self,
        action: str,
        id: str,
        enable_ml: bool,
        path: Optional[str],
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        if action == "push":
            if not path:
                raise ValueError("Path to file to push is required")
            return self._push(id, path, enable_ml)
        elif action == "pull":
            return self._pull(id)
        return ""

    async def _arun(
        self,
        action: str,
        id: str,
        enable_ml: bool,
        path: Optional[str] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("NucliaUnderstandingAPI does not support async")

    def _push(self, id: str, content_path: str, enable_ml: bool) -> str:
        with open(content_path, "rb") as source_file:
            response = requests.post(
                self._config["BACKEND"] + "/processing/upload",
                headers={
                    "content-type": mimetypes.guess_type(content_path)[0]
                    or "application/octet-stream",
                    "x-stf-nuakey": "Bearer " + self._config["NUA_KEY"],
                },
                data=source_file.read(),
            )
            if response.status_code != 200:
                print(
                    f"Error uploading {content_path}: "
                    f"{response.status_code} {response.text}"
                )
                return ""
            else:
                print(f"Pushing {content_path} in queue")
                file_data = {}
                file_data["file"] = f"{response.text}"
                response = requests.post(
                    self._config["BACKEND"] + "/processing/push",
                    headers={
                        "content-type": "application/json",
                        "x-stf-nuakey": "Bearer " + self._config["NUA_KEY"],
                    },
                    json={
                        "filefield": file_data,
                        "processing_options": {"ml_text": enable_ml},
                    },
                )
                if response.status_code != 200:
                    print(
                        f"Error pushing {content_path}: "
                        f"{response.status_code} {response.text}"
                    )
                    return ""
                else:
                    uuid = response.json()["uuid"]
                    print(f"Pushed {content_path} in queue, uuid: {uuid}")
                    self._results[id] = {"uuid": uuid, "status": "pending"}
                    return uuid

    def _pull(self, id: str) -> str:
        self._pull_queue()
        result = self._results.get(id, None)
        if not result:
            print(f"{id} not in queue")
            return ""
        elif result["status"] == "pending":
            print(f'Waiting for {result["uuid"]} to be processed')
            return ""
        else:
            return result["data"]

    def _pull_queue(self) -> None:
        res = requests.get(
            self._config["BACKEND"] + "/processing/pull",
            headers={
                "x-stf-nuakey": "Bearer " + self._config["NUA_KEY"],
            },
        ).json()
        if res["status"] == "empty":
            print("Queue empty")
        elif res["status"] == "ok":
            payload = res["payload"]
            pb = BrokerMessage()
            pb.ParseFromString(base64.b64decode(payload))
            uuid = pb.uuid
            print(f"Pulled {uuid} from queue")
            matching_id = self._find_matching_id(uuid)
            if not matching_id:
                print(f"No matching id for {uuid}")
            else:
                self._results[matching_id]["status"] = "done"
                data = MessageToJson(
                    pb,
                    preserving_proto_field_name=True,
                    including_default_value_fields=True,
                )
                self._results[matching_id]["data"] = data

    def _find_matching_id(self, uuid: str) -> Union[str, None]:
        for id, result in self._results.items():
            if result["uuid"] == uuid:
                return id
        return None
