"""Tool for the Nuclia Understanding API.

Installation:

```bash
    pip install --upgrade protobuf
    pip install nucliadb-protos
```
"""

import asyncio
import base64
import logging
import mimetypes
import os
from typing import Any, Dict, Optional, Type, Union

import requests
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class NUASchema(BaseModel):
    """Input for Nuclia Understanding API.

    Attributes:
        action: Action to perform. Either `push` or `pull`.
        id: ID of the file to push or pull.
        path: Path to the file to push (needed only for `push` action).
        text: Text content to process (needed only for `push` action).
    """

    action: str = Field(
        ...,
        description="Action to perform. Either `push` or `pull`.",
    )
    id: str = Field(
        ...,
        description="ID of the file to push or pull.",
    )
    path: Optional[str] = Field(
        ...,
        description="Path to the file to push (needed only for `push` action).",
    )
    text: Optional[str] = Field(
        ...,
        description="Text content to process (needed only for `push` action).",
    )


class NucliaUnderstandingAPI(BaseTool):
    """Tool to process files with the Nuclia Understanding API."""

    name: str = "nuclia_understanding_api"
    description: str = (
        "A wrapper around Nuclia Understanding API endpoints. "
        "Useful for when you need to extract text from any kind of files. "
    )
    args_schema: Type[BaseModel] = NUASchema
    _results: Dict[str, Any] = {}
    _config: Dict[str, Any] = {}

    def __init__(self, enable_ml: bool = False) -> None:
        zone = os.environ.get("NUCLIA_ZONE", "europe-1")
        self._config["BACKEND"] = f"https://{zone}.nuclia.cloud/api/v1"
        key = os.environ.get("NUCLIA_NUA_KEY")
        if not key:
            raise ValueError("NUCLIA_NUA_KEY environment variable not set")
        else:
            self._config["NUA_KEY"] = key
        self._config["enable_ml"] = enable_ml
        super().__init__()  # type: ignore[call-arg]

    def _run(
        self,
        action: str,
        id: str,
        path: Optional[str],
        text: Optional[str],
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        if action == "push":
            self._check_params(path, text)
            if path:
                return self._pushFile(id, path)
            if text:
                return self._pushText(id, text)
        elif action == "pull":
            return self._pull(id)
        return ""

    async def _arun(
        self,
        action: str,
        id: str,
        path: Optional[str] = None,
        text: Optional[str] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        self._check_params(path, text)
        if path:
            self._pushFile(id, path)
        if text:
            self._pushText(id, text)
        data = None
        while True:
            data = self._pull(id)
            if data:
                break
            await asyncio.sleep(15)
        return data

    def _pushText(self, id: str, text: str) -> str:
        field = {
            "textfield": {"text": {"body": text, "format": 0}},
            "processing_options": {"ml_text": self._config["enable_ml"]},
        }
        return self._pushField(id, field)

    def _pushFile(self, id: str, content_path: str) -> str:
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
                logger.info(
                    f"Error uploading {content_path}: "
                    f"{response.status_code} {response.text}"
                )
                return ""
            else:
                field = {
                    "filefield": {"file": f"{response.text}"},
                    "processing_options": {"ml_text": self._config["enable_ml"]},
                }
                return self._pushField(id, field)

    def _pushField(self, id: str, field: Any) -> str:
        logger.info(f"Pushing {id} in queue")
        response = requests.post(
            self._config["BACKEND"] + "/processing/push",
            headers={
                "content-type": "application/json",
                "x-stf-nuakey": "Bearer " + self._config["NUA_KEY"],
            },
            json=field,
        )
        if response.status_code != 200:
            logger.info(
                f"Error pushing field {id}:" f"{response.status_code} {response.text}"
            )
            raise ValueError("Error pushing field")
        else:
            uuid = response.json()["uuid"]
            logger.info(f"Field {id} pushed in queue, uuid: {uuid}")
            self._results[id] = {"uuid": uuid, "status": "pending"}
            return uuid

    def _pull(self, id: str) -> str:
        self._pull_queue()
        result = self._results.get(id, None)
        if not result:
            logger.info(f"{id} not in queue")
            return ""
        elif result["status"] == "pending":
            logger.info(f'Waiting for {result["uuid"]} to be processed')
            return ""
        else:
            return result["data"]

    def _pull_queue(self) -> None:
        try:
            from nucliadb_protos.writer_pb2 import BrokerMessage
        except ImportError as e:
            raise ImportError(
                "nucliadb-protos is not installed. "
                "Run `pip install nucliadb-protos` to install."
            ) from e
        try:
            from google.protobuf.json_format import MessageToJson
        except ImportError as e:
            raise ImportError(
                "Unable to import google.protobuf, please install with "
                "`pip install protobuf`."
            ) from e

        res = requests.get(
            self._config["BACKEND"] + "/processing/pull",
            headers={
                "x-stf-nuakey": "Bearer " + self._config["NUA_KEY"],
            },
        ).json()
        if res["status"] == "empty":
            logger.info("Queue empty")
        elif res["status"] == "ok":
            payload = res["payload"]
            pb = BrokerMessage()
            pb.ParseFromString(base64.b64decode(payload))
            uuid = pb.uuid
            logger.info(f"Pulled {uuid} from queue")
            matching_id = self._find_matching_id(uuid)
            if not matching_id:
                logger.info(f"No matching id for {uuid}")
            else:
                self._results[matching_id]["status"] = "done"
                data = MessageToJson(
                    pb,
                    preserving_proto_field_name=True,
                    including_default_value_fields=True,  # type: ignore
                )
                self._results[matching_id]["data"] = data

    def _find_matching_id(self, uuid: str) -> Union[str, None]:
        for id, result in self._results.items():
            if result["uuid"] == uuid:
                return id
        return None

    def _check_params(self, path: Optional[str], text: Optional[str]) -> None:
        if not path and not text:
            raise ValueError("File path or text is required")
        if path and text:
            raise ValueError("Cannot process both file and text on a single run")
