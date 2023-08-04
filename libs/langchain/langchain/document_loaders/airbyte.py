"""Loads local airbyte json files."""
from typing import Any, Callable, Iterator, List, Literal, Mapping, Optional

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

from libs.langchain.langchain.utils.utils import guard_import


RecordHandler = Callable[[Any, Optional[str]], Document]


class AirbyteCDKLoader(BaseLoader):
    """Loads records using an Airbyte source connector implemented using the CDK."""

    def __init__(
        self,
        config: Mapping[str, Any],
        source_class: Any,
        stream_name: str,
        record_handler: Optional[RecordHandler] = None,
        state: Optional[Any] = None,
    ) -> None:
        from airbyte_cdk.sources.embedded.base_integration import (
            BaseEmbeddedIntegration,
        )
        from airbyte_cdk.sources.embedded.runner import CDKRunner
        from airbyte_cdk.models.airbyte_protocol import AirbyteRecordMessage

        class CDKIntegration(BaseEmbeddedIntegration):
            def _handle_record(
                self, record: AirbyteRecordMessage, id: Optional[str]
            ) -> Document:
                if record_handler:
                    return record_handler(record, id)
                return Document(page_content="", metadata=record.data)

        self._integration = CDKIntegration(
            config=config,
            runner=CDKRunner(source=source_class(), name=source_class.__name__),
        )
        self._stream_name = stream_name
        self._state = state

    def load(self) -> List[Document]:
        return list(self.lazy_load())

    def lazy_load(self) -> Iterator[Document]:
        return self._integration._load_data(
            stream_name=self._stream_name, state=self._state
        )


class AirbyteHubspotLoader(AirbyteCDKLoader):
    def __init__(
        self,
        config: Mapping[str, Any],
        stream_name: str,
        record_handler: Optional[RecordHandler] = None,
        state: Optional[Any] = None,
    ) -> None:
        source_class = guard_import(
            "source_hubspot", "airbyte-source-hubspot"
        ).SourceHubspot
        super().__init__(
            config=config,
            source_class=source_class,
            stream_name=stream_name,
            record_handler=record_handler,
            state=state,
        )


class AirbyteStripeLoader(AirbyteCDKLoader):
    def __init__(
        self,
        config: Mapping[str, Any],
        stream_name: str,
        record_handler: Optional[RecordHandler] = None,
        state: Optional[Any] = None,
    ) -> None:
        source_class = guard_import(
            "source_stripe", "airbyte-source-stripe"
        ).SourceStripe
        super().__init__(
            config=config,
            source_class=source_class,
            stream_name=stream_name,
            record_handler=record_handler,
            state=state,
        )


class AirbyteTypeformLoader(AirbyteCDKLoader):
    def __init__(
        self,
        config: Mapping[str, Any],
        stream_name: str,
        record_handler: Optional[RecordHandler] = None,
        state: Optional[Any] = None,
    ) -> None:
        source_class = guard_import(
            "source_typeform", "airbyte-source-typeform"
        ).SourceTypeform
        super().__init__(
            config=config,
            source_class=source_class,
            stream_name=stream_name,
            record_handler=record_handler,
            state=state,
        )


class AirbyteZendeskSupportLoader(AirbyteCDKLoader):
    def __init__(
        self,
        config: Mapping[str, Any],
        stream_name: str,
        record_handler: Optional[RecordHandler] = None,
        state: Optional[Any] = None,
    ) -> None:
        source_class = guard_import(
            "source_zendesk_support", "airbyte-source-zendesk-support"
        ).SourceZendeskSupport
        super().__init__(
            config=config,
            source_class=source_class,
            stream_name=stream_name,
            record_handler=record_handler,
            state=state,
        )


class AirbyteShopifyLoader(AirbyteCDKLoader):
    def __init__(
        self,
        config: Mapping[str, Any],
        stream_name: str,
        record_handler: Optional[RecordHandler] = None,
        state: Optional[Any] = None,
    ) -> None:
        source_class = guard_import(
            "source_shopify", "airbyte-source-shopify"
        ).SourceShopify
        super().__init__(
            config=config,
            source_class=source_class,
            stream_name=stream_name,
            record_handler=record_handler,
            state=state,
        )


class AirbyteSalesforceLoader(AirbyteCDKLoader):
    def __init__(
        self,
        config: Mapping[str, Any],
        stream_name: str,
        record_handler: Optional[RecordHandler] = None,
        state: Optional[Any] = None,
    ) -> None:
        source_class = guard_import(
            "source_salesforce", "airbyte-source-salesforce"
        ).SourceSalesforce
        super().__init__(
            config=config,
            source_class=source_class,
            stream_name=stream_name,
            record_handler=record_handler,
            state=state,
        )
