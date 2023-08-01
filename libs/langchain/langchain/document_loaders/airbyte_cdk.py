"""Loads local airbyte json files."""
import json
from typing import Any, Iterator, List, Mapping, Optional

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from airbyte_protocol.models.airbyte_protocol import AirbyteRecordMessage, AirbyteStateMessage
from airbyte_cdk.sources.embedded.base_integration import BaseEmbeddedIntegration
from airbyte_cdk.sources.embedded.runner import CDKRunner


class AirbyteCDKLoader(BaseLoader, BaseEmbeddedIntegration):
    """Loads records using an Airbyte source connector implemented using the CDK."""

    def __init__(self, config: Mapping[str, Any], source_class: Any, stream_name: str, state: Optional[AirbyteStateMessage] = None) -> None:
        super().__init__(config=config, runner=CDKRunner(source=source_class(), name=source_class.__name__))
        self._stream_name = stream_name
        self._state = state

    def _handle_record(self, record: AirbyteRecordMessage, id: Optional[str]) -> Document:
        return Document(page_content="", extra_info=record.data)

    def load(self) -> List[Document]:
        return list(self._load_data(stream_name=self._stream_name, state=self._state))

    def lazy_load(self) -> Iterator[Document]:
        return self._load_data(stream_name=self._stream_name, state=self._state)