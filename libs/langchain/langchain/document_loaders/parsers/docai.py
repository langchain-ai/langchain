"""Module contains a PDF parser based on DocAI from Google Cloud.

You need to install two libraries to use this parser:
pip install google-cloud-documentai
pip install google-cloud-documentai-toolbox
"""
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, List, Optional, Sequence

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseBlobParser
from langchain.document_loaders.blob_loaders import Blob
from langchain.utils.iter import batch_iterate

if TYPE_CHECKING:
    from google.api_core.operation import Operation
    from google.cloud.documentai import DocumentProcessorServiceClient


logger = logging.getLogger(__name__)


@dataclass
class DocAIParsingResults:
    """A dataclass to store DocAI parsing results."""

    source_path: str
    parsed_path: str


class DocAIParser(BaseBlobParser):
    def __init__(
        self,
        *,
        client: Optional["DocumentProcessorServiceClient"] = None,
        location: Optional[str] = None,
        gcs_output_path: Optional[str] = None,
        processor_name: Optional[str] = None,
    ):
        """Initializes the parser.

        Args:
            client: a DocumentProcessorServiceClient to use
            location: a GCP location where a DOcAI parser is located
            gcs_output_path: a path on GCS to store parsing results
            processor_name: name of a processor

        You should provide either a client or location (and then a client
            would be instantiated).
        """
        if client and location:
            raise ValueError(
                "You should provide either a client or a location but not both "
                "of them."
            )
        if not client and not location:
            raise ValueError(
                "You must specify either a client or a location to instantiate "
                "a client."
            )

        self._gcs_output_path = gcs_output_path
        self._processor_name = processor_name
        if client:
            self._client = client
        else:
            try:
                from google.api_core.client_options import ClientOptions
                from google.cloud.documentai import DocumentProcessorServiceClient
            except ImportError:
                raise ImportError(
                    "documentai package not found, please install it with"
                    " `pip install google-cloud-documentai`"
                )
            options = ClientOptions(
                api_endpoint=f"{location}-documentai.googleapis.com"
            )
            self._client = DocumentProcessorServiceClient(client_options=options)

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Parses a blob lazily.

        Args:
            blobs: a Blob to parse

        This is a long-running operations! A recommended way is to batch
            documents together and use `batch_parse` method.
        """
        yield from self.batch_parse([blob], gcs_output_path=self._gcs_output_path)

    def batch_parse(
        self,
        blobs: Sequence[Blob],
        gcs_output_path: Optional[str] = None,
        timeout_sec: int = 3600,
        check_in_interval_sec: int = 60,
    ) -> Iterator[Document]:
        """Parses a list of blobs lazily.

        Args:
            blobs: a list of blobs to parse
            gcs_output_path: a path on GCS to store parsing results
            timeout_sec: a timeout to wait for DocAI to complete, in seconds
            check_in_interval_sec: an interval to wait until next check
                whether parsing operations have been completed, in seconds
        This is a long-running operations! A recommended way is to decouple
            parsing from creating Langchain Documents:
            >>> operations = parser.docai_parse(blobs, gcs_path)
            >>> parser.is_running(operations)
            You can get operations names and save them:
            >>> names = [op.operation.name for op in operations]
            And when all operations are finished, you can use their results:
            >>> operations = parser.operations_from_names(operation_names)
            >>> results = parser.get_results(operations)
            >>> docs = parser.parse_from_results(results)
        """
        output_path = gcs_output_path if gcs_output_path else self._gcs_output_path
        if output_path is None:
            raise ValueError("An output path on GCS should be provided!")
        operations = self.docai_parse(blobs, gcs_output_path=output_path)
        operation_names = [op.operation.name for op in operations]
        logger.debug(
            f"Started parsing with DocAI, submitted operations {operation_names}"
        )
        is_running, time_elapsed = True, 0
        while is_running:
            is_running = self.is_running(operations)
            if not is_running:
                break
            time.sleep(check_in_interval_sec)
            time_elapsed += check_in_interval_sec
            if time_elapsed > timeout_sec:
                raise ValueError(
                    "Timeout exceeded! Check operations " f"{operation_names} later!"
                )
            logger.debug(".")

        results = self.get_results(operations=operations)
        yield from self.parse_from_results(results)

    def parse_from_results(
        self, results: List[DocAIParsingResults]
    ) -> Iterator[Document]:
        try:
            from google.cloud.documentai_toolbox.wrappers.document import _get_shards
            from google.cloud.documentai_toolbox.wrappers.page import _text_from_layout
        except ImportError:
            raise ImportError(
                "documentai_toolbox package not found, please install it with"
                " `pip install google-cloud-documentai-toolbox`"
            )
        for result in results:
            output_gcs = result.parsed_path.split("/")
            gcs_bucket_name = output_gcs[2]
            gcs_prefix = "/".join(output_gcs[3:]) + "/"
            shards = _get_shards(gcs_bucket_name, gcs_prefix)
            docs, page_number = [], 1
            for shard in shards:
                for page in shard.pages:
                    docs.append(
                        Document(
                            page_content=_text_from_layout(page.layout, shard.text),
                            metadata={
                                "page": page_number,
                                "source": result.source_path,
                            },
                        )
                    )
                    page_number += 1
            yield from docs

    def operations_from_names(self, operation_names: List[str]) -> List["Operation"]:
        """Initializes Long-Running Operations from their names."""
        try:
            from google.longrunning.operations_pb2 import (
                GetOperationRequest,  # type: ignore
            )
        except ImportError:
            raise ImportError(
                "documentai package not found, please install it with"
                " `pip install gapic-google-longrunning`"
            )

        operations = []
        for name in operation_names:
            request = GetOperationRequest(name=name)
            operations.append(self._client.get_operation(request=request))
        return operations

    def is_running(self, operations: List["Operation"]) -> bool:
        for op in operations:
            if not op.done():
                return True
        return False

    def docai_parse(
        self,
        blobs: Sequence[Blob],
        *,
        gcs_output_path: Optional[str] = None,
        batch_size: int = 4000,
        enable_native_pdf_parsing: bool = True,
    ) -> List["Operation"]:
        """Runs Google DocAI PDF parser on a list of blobs.

        Args:
            blobs: a list of blobs to be parsed
            gcs_output_path: a path (folder) on GCS to store results
            batch_size: amount of documents per batch
            enable_native_pdf_parsing: a config option for the parser

        DocAI has a limit on the amount of documents per batch, that's why split a
            batch into mini-batches. Parsing is an async long-running operation
            on Google Cloud and results are stored in a output GCS bucket.
        """
        try:
            from google.cloud import documentai
            from google.cloud.documentai_v1.types import OcrConfig, ProcessOptions
        except ImportError:
            raise ImportError(
                "documentai package not found, please install it with"
                " `pip install google-cloud-documentai`"
            )

        if not self._processor_name:
            raise ValueError("Processor name is not defined, aborting!")
        output_path = gcs_output_path if gcs_output_path else self._gcs_output_path
        if output_path is None:
            raise ValueError("An output path on GCS should be provided!")

        operations = []
        for batch in batch_iterate(size=batch_size, iterable=blobs):
            documents = []
            for blob in batch:
                gcs_document = documentai.GcsDocument(
                    gcs_uri=blob.path, mime_type="application/pdf"
                )
                documents.append(gcs_document)
            gcs_documents = documentai.GcsDocuments(documents=documents)

            input_config = documentai.BatchDocumentsInputConfig(
                gcs_documents=gcs_documents
            )

            gcs_output_config = documentai.DocumentOutputConfig.GcsOutputConfig(
                gcs_uri=output_path, field_mask=None
            )
            output_config = documentai.DocumentOutputConfig(
                gcs_output_config=gcs_output_config
            )

            if enable_native_pdf_parsing:
                process_options = ProcessOptions(
                    ocr_config=OcrConfig(
                        enable_native_pdf_parsing=enable_native_pdf_parsing
                    )
                )
            else:
                process_options = ProcessOptions()
            request = documentai.BatchProcessRequest(
                name=self._processor_name,
                input_documents=input_config,
                document_output_config=output_config,
                process_options=process_options,
            )
            operations.append(self._client.batch_process_documents(request))
        return operations

    def get_results(self, operations: List["Operation"]) -> List[DocAIParsingResults]:
        try:
            from google.cloud.documentai_v1 import BatchProcessMetadata
        except ImportError:
            raise ImportError(
                "documentai package not found, please install it with"
                " `pip install google-cloud-documentai`"
            )

        results = []
        for op in operations:
            if isinstance(op.metadata, BatchProcessMetadata):
                metadata = op.metadata
            else:
                metadata = BatchProcessMetadata.deserialize(op.metadata.value)
            for status in metadata.individual_process_statuses:
                source = status.input_gcs_source
                output = status.output_gcs_destination
                results.append(
                    DocAIParsingResults(source_path=source, parsed_path=output)
                )
        return results
