from typing import Any, Callable, Iterator, List, Mapping, Optional

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.utils.utils import guard_import

RecordHandler = Callable[[Any, Optional[str]], Document]


class AirbyteCDKLoader(BaseLoader):
    """Load with an `Airbyte` source connector implemented using the `CDK`."""

    def __init__(
        self,
        config: Mapping[str, Any],
        source_class: Any,
        stream_name: str,
        record_handler: Optional[RecordHandler] = None,
        state: Optional[Any] = None,
    ) -> None:
        """Initializes the loader.

        Args:
            config: The config to pass to the source connector.
            source_class: The source connector class.
            stream_name: The name of the stream to load.
            record_handler: A function that takes in a record and an optional id and
                returns a Document. If None, the record will be used as the document.
                Defaults to None.
            state: The state to pass to the source connector. Defaults to None.
        """
        from airbyte_cdk.models.airbyte_protocol import AirbyteRecordMessage
        from airbyte_cdk.sources.embedded.base_integration import (
            BaseEmbeddedIntegration,
        )
        from airbyte_cdk.sources.embedded.runner import CDKRunner

        class CDKIntegration(BaseEmbeddedIntegration):
            """A wrapper around the CDK integration."""

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

    @property
    def last_state(self) -> Any:
        return self._integration.last_state


class AirbyteHubspotLoader(AirbyteCDKLoader):
    """Load from `Hubspot` using an `Airbyte` source connector."""

    def __init__(
        self,
        config: Mapping[str, Any],
        stream_name: str,
        record_handler: Optional[RecordHandler] = None,
        state: Optional[Any] = None,
    ) -> None:
        """Initializes the loader.

        Args:
            config: The config to pass to the source connector.
            stream_name: The name of the stream to load.
            record_handler: A function that takes in a record and an optional id and
                returns a Document. If None, the record will be used as the document.
                Defaults to None.
            state: The state to pass to the source connector. Defaults to None.
        """
        source_class = guard_import(
            "source_hubspot", pip_name="airbyte-source-hubspot"
        ).SourceHubspot
        super().__init__(
            config=config,
            source_class=source_class,
            stream_name=stream_name,
            record_handler=record_handler,
            state=state,
        )


class AirbyteStripeLoader(AirbyteCDKLoader):
    """Load from `Stripe` using an `Airbyte` source connector."""

    def __init__(
        self,
        config: Mapping[str, Any],
        stream_name: str,
        record_handler: Optional[RecordHandler] = None,
        state: Optional[Any] = None,
    ) -> None:
        """Initializes the loader.

        Args:
            config: The config to pass to the source connector.
            stream_name: The name of the stream to load.
            record_handler: A function that takes in a record and an optional id and
                returns a Document. If None, the record will be used as the document.
                Defaults to None.
            state: The state to pass to the source connector. Defaults to None.
        """
        source_class = guard_import(
            "source_stripe", pip_name="airbyte-source-stripe"
        ).SourceStripe
        super().__init__(
            config=config,
            source_class=source_class,
            stream_name=stream_name,
            record_handler=record_handler,
            state=state,
        )


class AirbyteTypeformLoader(AirbyteCDKLoader):
    """Load from `Typeform` using an `Airbyte` source connector."""

    def __init__(
        self,
        config: Mapping[str, Any],
        stream_name: str,
        record_handler: Optional[RecordHandler] = None,
        state: Optional[Any] = None,
    ) -> None:
        """Initializes the loader.

        Args:
            config: The config to pass to the source connector.
            stream_name: The name of the stream to load.
            record_handler: A function that takes in a record and an optional id and
                returns a Document. If None, the record will be used as the document.
                Defaults to None.
            state: The state to pass to the source connector. Defaults to None.
        """
        source_class = guard_import(
            "source_typeform", pip_name="airbyte-source-typeform"
        ).SourceTypeform
        super().__init__(
            config=config,
            source_class=source_class,
            stream_name=stream_name,
            record_handler=record_handler,
            state=state,
        )


class AirbyteZendeskSupportLoader(AirbyteCDKLoader):
    """Load from `Zendesk Support` using an `Airbyte` source connector."""

    def __init__(
        self,
        config: Mapping[str, Any],
        stream_name: str,
        record_handler: Optional[RecordHandler] = None,
        state: Optional[Any] = None,
    ) -> None:
        """Initializes the loader.

        Args:
            config: The config to pass to the source connector.
            stream_name: The name of the stream to load.
            record_handler: A function that takes in a record and an optional id and
                returns a Document. If None, the record will be used as the document.
                Defaults to None.
            state: The state to pass to the source connector. Defaults to None.
        """
        source_class = guard_import(
            "source_zendesk_support", pip_name="airbyte-source-zendesk-support"
        ).SourceZendeskSupport
        super().__init__(
            config=config,
            source_class=source_class,
            stream_name=stream_name,
            record_handler=record_handler,
            state=state,
        )


class AirbyteShopifyLoader(AirbyteCDKLoader):
    """Load from `Shopify` using an `Airbyte` source connector."""

    def __init__(
        self,
        config: Mapping[str, Any],
        stream_name: str,
        record_handler: Optional[RecordHandler] = None,
        state: Optional[Any] = None,
    ) -> None:
        """Initializes the loader.

        Args:
            config: The config to pass to the source connector.
            stream_name: The name of the stream to load.
            record_handler: A function that takes in a record and an optional id and
                returns a Document. If None, the record will be used as the document.
                Defaults to None.
            state: The state to pass to the source connector. Defaults to None.
        """
        source_class = guard_import(
            "source_shopify", pip_name="airbyte-source-shopify"
        ).SourceShopify
        super().__init__(
            config=config,
            source_class=source_class,
            stream_name=stream_name,
            record_handler=record_handler,
            state=state,
        )


class AirbyteSalesforceLoader(AirbyteCDKLoader):
    """Load from `Salesforce` using an `Airbyte` source connector."""

    def __init__(
        self,
        config: Mapping[str, Any],
        stream_name: str,
        record_handler: Optional[RecordHandler] = None,
        state: Optional[Any] = None,
    ) -> None:
        """Initializes the loader.

        Args:
            config: The config to pass to the source connector.
            stream_name: The name of the stream to load.
            record_handler: A function that takes in a record and an optional id and
                returns a Document. If None, the record will be used as the document.
                Defaults to None.
            state: The state to pass to the source connector. Defaults to None.
        """
        source_class = guard_import(
            "source_salesforce", pip_name="airbyte-source-salesforce"
        ).SourceSalesforce
        super().__init__(
            config=config,
            source_class=source_class,
            stream_name=stream_name,
            record_handler=record_handler,
            state=state,
        )


class AirbyteGongLoader(AirbyteCDKLoader):
    """Load from `Gong` using an `Airbyte` source connector."""

    def __init__(
        self,
        config: Mapping[str, Any],
        stream_name: str,
        record_handler: Optional[RecordHandler] = None,
        state: Optional[Any] = None,
    ) -> None:
        """Initializes the loader.

        Args:
            config: The config to pass to the source connector.
            stream_name: The name of the stream to load.
            record_handler: A function that takes in a record and an optional id and
                returns a Document. If None, the record will be used as the document.
                Defaults to None.
            state: The state to pass to the source connector. Defaults to None.
        """
        source_class = guard_import(
            "source_gong", pip_name="airbyte-source-gong"
        ).SourceGong
        super().__init__(
            config=config,
            source_class=source_class,
            stream_name=stream_name,
            record_handler=record_handler,
            state=state,
        )
