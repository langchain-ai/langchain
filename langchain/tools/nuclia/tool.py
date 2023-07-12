"""Tool for the Nuclia Understanding API.

Installation:

```bash
    pip install --upgrade protobuf
    pip install nucliadb-protos
```
"""

import base64
import json
import mimetypes
import os
import requests
import urllib3

from typing import Any, Dict, Optional, Type
from pydantic import BaseModel, Field

from google.protobuf.json_format import MessageToDict
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

urllib3.disable_warnings()

class NUASchema(BaseModel):
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


class NucliaUnderstandingAPI(BaseTool):
    """Tool to process files with the Nuclia Understanding API."""

    name = "nuclia_understanding_api"
    description = (
        "A wrapper around Nuclia Understanding API endpoints. "
        "Useful for when you need to extract text from any kind of files. "
    )
    args_schema: Type[BaseModel] = NUASchema
    _results: Dict[str, Any] = {}
 
    def _run(
        self,
        action: str,
        id: str,
        path: Optional[str],
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        zone = os.environ.get("NUCLIA_ZONE", "europe-1")
        BACKEND = f"https://{zone}.nuclia.cloud/api/v1"
        NUA_KEY = os.environ.get("NUCLIA_NUA_KEY")
        if not NUA_KEY:
            raise ValueError("NUCLIA_NUA_KEY environment variable not set")
        if action == "push":
            if not path:
                raise ValueError("Path to file to push is required")
            return self._push(id, path, BACKEND, NUA_KEY)
        elif action == "pull":
            return self._pull(id, BACKEND, NUA_KEY)

    async def _arun(
        self,
        action: str,
        id: str,
        path: Optional[str] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("NucliaUnderstandingAPI does not support async")

    def _push(self, id, content_path, backend, key):
        with open(content_path, "rb") as source_file:
            response = requests.post(
                f'{backend}/processing/upload',
                headers={
                    "content-type": mimetypes.guess_type(content_path)[0] or "application/octet-stream",
                    "x-stf-nuakey": "Bearer " + key,
                },
                data=source_file.read(),
                verify=False,
            )
            if response.status_code != 200:
                print(f'Error uploading {content_path}: {response.status_code} {response.text}')
            else:
                print(f'Pushing {content_path} in queue')
                file_data = {}
                file_data['file'] = f'{response.text}'
                response = requests.post(
                    f'{backend}/processing/push',
                    headers={
                        "content-type": "application/json",
                        "x-stf-nuakey": "Bearer " + key,
                    },
                    json={"filefield": file_data},
                    verify=False,
                )
                if response.status_code != 200:
                    print(f'Error pushing {content_path}: {response.status_code} {response.text}')
                else:
                    uuid = response.json()["uuid"]
                    print(f'Pushed {content_path} in queue, uuid: {uuid}')
                    self._results[id] = {"uuid": uuid, "status": "pending"}
                    return uuid
    
    def _pull(self, id, backend, key):
        self._pull_queue(backend, key)
        result = self._results.get(id, None)
        if not result:
            print(f'{id} not in queue')
            return None
        elif result['status'] == 'pending':
            print(f'Waiting for {result["uuid"]} to be processed')
            return None
        else:
            return result['data']

    def _pull_queue(self, backend, key):
        res = requests.get(
            f'{backend}/processing/pull',
            headers={
                "x-stf-nuakey": "Bearer " + key,
            },
            verify=False,
        ).json()
        if res['status'] == 'empty':
            print('Queue empty')
        elif res['status'] == 'ok':
            payload = res['payload']
            pb = BrokerMessage()
            pb.ParseFromString(base64.b64decode(payload))
            uuid = pb.uuid
            print(f'Pulled {uuid} from queue')
            matching_id = self._find_matching_id(uuid)
            if not matching_id:
                print(f'No matching id for {uuid}')
            else:
                self._results[matching_id]['status'] = 'done'
                data = MessageToDict(pb, preserving_proto_field_name=True, including_default_value_fields=True)
                self._results[matching_id]['data'] = data
            
    def _find_matching_id(self, uuid):
        for id, result in self._results.items():
            if result['uuid'] == uuid:
                return id
        return None