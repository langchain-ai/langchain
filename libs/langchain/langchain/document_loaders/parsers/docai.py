"""Module contains a PDF parser based on Document AI from Google Cloud.

You need to install two libraries to use this parser:
pip install google-cloud-documentai
pip install google-cloud-documentai-toolbox
"""
import logging
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, List, Optional, Sequence

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseBlobParser
from langchain.document_loaders.blob_loaders import Blob
from langchain.utilities.vertexai import get_client_info
from langchain.utils.iter import batch_iterate

if TYPE_CHECKING:
    from google.api_core.operation import Operation
    from google.cloud.documentai import DocumentProcessorServiceClient


logger = logging.getLogger(__name__)


@dataclass
class DocAIParsingResults:
    """A dataclass to store Document AI parsing results."""

    source_path: str
    parsed_path: str


class DocAIParser(BaseBlobParser):
    """`Google Cloud Document AI` parser.

    For a detailed explanation of Document AI, refer to the product documentation.
    https://cloud.google.com/document-ai/docs/overview
    """

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
            location: a Google Cloud location where a Document AI processor is located
            gcs_output_path: a path on Google Cloud Storage to store parsing results
            processor_name: full resource name of a Document AI processor or processor
                version

        You should provide either a client or location (and then a client
            would be instantiated).
        """

        if bool(client) == bool(location):
            raise ValueError(
                "You must specify either a client or a location to instantiate "
                "a client."
            )

        pattern = r"projects\/[0-9]+\/locations\/[a-z\-0-9]+\/processors\/[a-z0-9]+"
        if processor_name and not re.fullmatch(pattern, processor_name):
            raise ValueError(
                f"Processor name {processor_name} has the wrong format. If your "
                "prediction endpoint looks like https://us-documentai.googleapis.com"
                "/v1/projects/PROJECT_ID/locations/us/processors/PROCESSOR_ID:process,"
                " use only projects/PROJECT_ID/locations/us/processors/PROCESSOR_ID "
                "part."
            )

        self._gcs_output_path = gcs_output_path
        self._processor_name = processor_name
        if client:
            self._client = client
        else:
            try:
                from google.api_core.client_options import ClientOptions
                from google.cloud.documentai import DocumentProcessorServiceClient
            except ImportError as exc:
                raise ImportError(
                    "documentai package not found, please install it with"
                    " `pip install google-cloud-documentai`"
                ) from exc
            options = ClientOptions(
                api_endpoint=f"{location}-documentai.googleapis.com"
            )
            self._client = DocumentProcessorServiceClient(
                client_options=options,
                client_info=get_client_info(module="document-ai"),
            )

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Parses a blob lazily.

        Args:
            blobs: a Blob to parse

        This is a long-running operation. A recommended way is to batch
            documents together and use the `batch_parse()` method.
        """
        yield from self.batch_parse([blob], gcs_output_path=self._gcs_output_path)

    def online_process(
        self,
        blob: Blob,
        enable_native_pdf_parsing: bool = True,
        field_mask: Optional[str] = None,
        page_range: Optional[List[int]] = None,
    ) -> Iterator[Document]:
        """Parses a blob lazily using online processing.

        Args:
            blob: a blob to parse.
            enable_native_pdf_parsing: enable pdf embedded text extraction
            field_mask: a comma-separated list of which fields to include in the
                Document AI response.
                suggested: "text,pages.pageNumber,pages.layout"
            page_range: list of page numbers to parse. If `None`,
                entire document will be parsed.
        """
        try:
            from google.cloud import documentai
            from google.cloud.documentai_v1.types import (
                IndividualPageSelector,
                OcrConfig,
                ProcessOptions,
            )
        except ImportError as exc:
            raise ImportError(
                "documentai package not found, please install it with"
                " `pip install google-cloud-documentai`"
            ) from exc
        try:
            from google.cloud.documentai_toolbox.wrappers.page import _text_from_layout
        except ImportError as exc:
            raise ImportError(
                "documentai_toolbox package not found, please install it with"
                " `pip install google-cloud-documentai-toolbox`"
            ) from exc
        ocr_config = (
            OcrConfig(enable_native_pdf_parsing=enable_native_pdf_parsing)
            if enable_native_pdf_parsing
            else None
        )
        individual_page_selector = (
            IndividualPageSelector(pages=page_range) if page_range else None
        )

        response = self._client.process_document(
            documentai.ProcessRequest(
                name=self._processor_name,
                gcs_document=documentai.GcsDocument(
                    gcs_uri=blob.path,
                    mime_type=blob.mimetype or "application/pdf",
                ),
                process_options=ProcessOptions(
                    ocr_config=ocr_config,
                    individual_page_selector=individual_page_selector,
                ),
                skip_human_review=True,
                field_mask=field_mask,
            )
        )
        yield from (
            Document(
                page_content=_text_from_layout(page.layout, response.document.text),
                metadata={
                    "page": page.page_number,
                    "source": blob.path,
                },
            )
            for page in response.document.pages
        )

    def batch_parse(
        self,
        blobs: Sequence[Blob],
        gcs_output_path: Optional[str] = None,
        timeout_sec: int = 3600,
        check_in_interval_sec: int = 60,
    ) -> Iterator[Document]:
        """Parses a list of blobs lazily.

        Args:
            blobs: a list of blobs to parse.
            gcs_output_path: a path on Google Cloud Storage to store parsing results.
            timeout_sec: a timeout to wait for Document AI to complete, in seconds.
            check_in_interval_sec: an interval to wait until next check
                whether parsing operations have been completed, in seconds
        This is a long-running operation. A recommended way is to decouple
            parsing from creating LangChain Documents:
            >>> operations = parser.docai_parse(blobs, gcs_path)
            >>> parser.is_running(operations)
            You can get operations names and save them:
            >>> names = [op.operation.name for op in operations]
            And when all operations are finished, you can use their results:
            >>> operations = parser.operations_from_names(operation_names)
            >>> results = parser.get_results(operations)
            >>> docs = parser.parse_from_results(results)
        """
        output_path = gcs_output_path or self._gcs_output_path
        if not output_path:
            raise ValueError(
                "An output path on Google Cloud Storage should be provided."
            )
        operations = self.docai_parse(blobs, gcs_output_path=output_path)
        operation_names = [op.operation.name for op in operations]
        logger.debug(
            "Started parsing with Document AI, submitted operations %s", operation_names
        )
        time_elapsed = 0
        while self.is_running(operations):
            time.sleep(check_in_interval_sec)
            time_elapsed += check_in_interval_sec
            if time_elapsed > timeout_sec:
                raise TimeoutError(
                    "Timeout exceeded! Check operations " f"{operation_names} later!"
                )
            logger.debug(".")

        results = self.get_results(operations=operations)
        yield from self.parse_from_results(results)

    def parse_from_results(
        self, results: List[DocAIParsingResults]
    ) -> Iterator[Document]:
        try:
            from google.cloud.documentai_toolbox.utilities.gcs_utilities import (
                split_gcs_uri,
            )
            from google.cloud.documentai_toolbox.wrappers.document import _get_shards
            from google.cloud.documentai_toolbox.wrappers.page import _text_from_layout
        except ImportError as exc:
            raise ImportError(
                "documentai_toolbox package not found, please install it with"
                " `pip install google-cloud-documentai-toolbox`"
            ) from exc
        for result in results:
            gcs_bucket_name, gcs_prefix = split_gcs_uri(result.parsed_path)
            shards = _get_shards(gcs_bucket_name, gcs_prefix)
            yield from (
                Document(
                    page_content=_text_from_layout(page.layout, shard.text),
                    metadata={"page": page.page_number, "source": result.source_path},
                )
                for shard in shards
                for page in shard.pages
            )

    def operations_from_names(self, operation_names: List[str]) -> List["Operation"]:
        """Initializes Long-Running Operations from their names."""
        try:
            from google.longrunning.operations_pb2 import (
                GetOperationRequest,  # type: ignore
            )
        except ImportError as exc:
            raise ImportError(
                "long running operations package not found, please install it with"
                " `pip install gapic-google-longrunning`"
            ) from exc

        return [
            self._client.get_operation(request=GetOperationRequest(name=name))
            for name in operation_names
        ]

    def is_running(self, operations: List["Operation"]) -> bool:
        return any(not op.done() for op in operations)

    def docai_parse(
        self,
        blobs: Sequence[Blob],
        *,
        gcs_output_path: Optional[str] = None,
        processor_name: Optional[str] = None,
        batch_size: int = 1000,
        enable_native_pdf_parsing: bool = True,
        field_mask: Optional[str] = None,
    ) -> List["Operation"]:
        """Runs Google Document AI PDF Batch Processing on a list of blobs.

        Args:
            blobs: a list of blobs to be parsed
            gcs_output_path: a path (folder) on GCS to store results
            processor_name: name of a Document AI processor.
            batch_size: amount of documents per batch
            enable_native_pdf_parsing: a config option for the parser
            field_mask: a comma-separated list of which fields to include in the
                Document AI response.
                suggested: "text,pages.pageNumber,pages.layout"

        Document AI has a 1000 file limit per batch, so batches larger than that need
        to be split into multiple requests.
        Batch processing is an async long-running operation
        and results are stored in a output GCS bucket.
        """
        try:
            from google.cloud import documentai
            from google.cloud.documentai_v1.types import OcrConfig, ProcessOptions
        except ImportError as exc:
            raise ImportError(
                "documentai package not found, please install it with"
                " `pip install google-cloud-documentai`"
            ) from exc

        output_path = gcs_output_path or self._gcs_output_path
        if output_path is None:
            raise ValueError(
                "An output path on Google Cloud Storage should be provided."
            )
        processor_name = processor_name or self._processor_name
        if processor_name is None:
            raise ValueError("A Document AI processor name should be provided.")

        operations = []
        for batch in batch_iterate(size=batch_size, iterable=blobs):
            input_config = documentai.BatchDocumentsInputConfig(
                gcs_documents=documentai.GcsDocuments(
                    documents=[
                        documentai.GcsDocument(
                            gcs_uri=blob.path,
                            mime_type=blob.mimetype or "application/pdf",
                        )
                        for blob in batch
                    ]
                )
            )

            output_config = documentai.DocumentOutputConfig(
                gcs_output_config=documentai.DocumentOutputConfig.GcsOutputConfig(
                    gcs_uri=output_path, field_mask=field_mask
                )
            )

            process_options = (
                ProcessOptions(
                    ocr_config=OcrConfig(
                        enable_native_pdf_parsing=enable_native_pdf_parsing
                    )
                )
                if enable_native_pdf_parsing
                else None
            )
            operations.append(
                self._client.batch_process_documents(
                    documentai.BatchProcessRequest(
                        name=processor_name,
                        input_documents=input_config,
                        document_output_config=output_config,
                        process_options=process_options,
                        skip_human_review=True,
                    )
                )
            )
        return operations

    def get_results(self, operations: List["Operation"]) -> List[DocAIParsingResults]:
        try:
            from google.cloud.documentai_v1 import BatchProcessMetadata
        except ImportError as exc:
            raise ImportError(
                "documentai package not found, please install it with"
                " `pip install google-cloud-documentai`"
            ) from exc

        return [
            DocAIParsingResults(
                source_path=status.input_gcs_source,
                parsed_path=status.output_gcs_destination,
            )
            for op in operations
            for status in (
                op.metadata.individual_process_statuses
                if isinstance(op.metadata, BatchProcessMetadata)
                else BatchProcessMetadata.deserialize(
                    op.metadata.value
                ).individual_process_statuses
            )
        ]
