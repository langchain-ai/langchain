"""Loads local airbyte json files."""
import json
from typing import Any, Mapping, Optional

from langchain.document_loaders.airbyte_cdk import AirbyteCDKLoader
from airbyte_protocol.models.airbyte_protocol import AirbyteStateMessage


class AirbyteHubspotLoader(AirbyteCDKLoader):
    def __init__(self, config: Mapping[str, Any], stream_name: str, state: Optional[AirbyteStateMessage] = None) -> None:
        import source_hubspot
        super().__init__(config=config, source_class=source_hubspot.SourceHubspot, stream_name=stream_name, state=state)
