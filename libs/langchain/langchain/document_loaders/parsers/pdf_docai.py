"""Module contains a PDF parser based on DocAI from Google Cloud.

You need to install two libraries to use this parser:
pip install google-cloud-documentai
pip install google-cloud-documentai-toolbox
"""
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, List, Optional

from google.api_core.operation import Operation

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseBlobParser
from langchain.document_loaders.blob_loaders import Blob

if TYPE_CHECKING:
    from google.cloud.documentai import DocumentProcessorServiceClient


@dataclass
class DocAIParsingResults:
    """A dataclass to store DocAI parsing results."""

    source_path: str
    parsed_path: str


class DocAIPdfLoader(BaseBlobParser):
    def __init__(
        self,
        project: str,
        *,
        location: str = "us",
        processor_name: Optional[str] = None,
        gcs_output_path: Optional[str] = None,
    ):
        from google.api_core.client_options import ClientOptions

        self._options = ClientOptions(
            api_endpoint=f"{location}-documentai.googleapis.com"
        )
        self._processor_name = processor_name
        self._name = self.client.common_location_path(project=project, location="us")
        self._gcs_output_path = gcs_output_path

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Parses a blob lazily.

        This is a long-running operations! A recommended way is to batch
            documents together and use `batch_parse` method.
        """
        yield from self.batch_parse(
            [blob], self._gcs_output_path  # type: ignore[arg-type]
        )

    def batch_parse(
        self, blobs: Sequence[Blob], gcs_output_path: str
    ) -> Iterator[Document]:
        """Parses a list of blobs lazily.

        Args:
            blobs: a list of blobs to parse
            gcs_output_path: a path on GCS to store parsing results
        This is a long-running operations! A recommended way is to decouple
            parsing from creating Langchain Documents:
            >>> operations = parser.docai_parse(blobs, gcs_path)
            >>> parser.is_running(operations)
            You can get operations names and save them:
            >>> names = [op.name for op in operations]
            And when all operations are finished, you can use their results:
            >>> operations = parser.operations_from_names(operation_names)
            >>> results = parser.get_results(operations)
            >>> docs = parser.batch_parse_from_results(results)
        """
        operations = self.docai_parse(blobs, gcs_output_path=gcs_output_path)
        is_running = self.is_running(operations)
        while is_running:
            is_running = self.is_running(operations)
            if not is_running:
                break
            time.sleep(60)
            print(".", end="")

        results = self.get_results(operations=operations)
        yield from self.batch_parse_from_results(results)

    def batch_parse_from_results(
        self, results: List[DocAIParsingResults]
    ) -> Iterator[Document]:
        for result in results:
            for doc in self.parse_from_result(result):
                yield doc

    def parse_from_result(self, result: DocAIParsingResults) -> List[Document]:
        """Creates a list of Documents from Google DocAI parsing results."""
        from google.cloud.documentai_toolbox.wrappers.document import _get_shards
        from google.cloud.documentai_toolbox.wrappers.page import _text_from_layout

        output_gcs = result.parsed_path.split("/")
        gcs_bucket_name = output_gcs[2]
        gcs_prefix = "/".join(output_gcs[3:]) + "/"
        shards = _get_shards(gcs_bucket_name, gcs_prefix)
        docs, i = [], 0
        for shard in shards:
            for page in shard.pages:
                docs.append(
                    Document(
                        page_content=_text_from_layout(page.layout, shard.text),
                        metadata={"page": i + 1, "source": result.source_path},
                    )
                )
                i += 1
        return docs

    @property
    def client(self) -> "DocumentProcessorServiceClient":
        """Returns a docai client to be used by the parser."""
        from google.cloud.documentai import DocumentProcessorServiceClient

        return DocumentProcessorServiceClient(client_options=self._options)

    def operations_from_names(self, operation_names: List[str]) -> List[Operation]:
        """Initializes Long-Running Operations from their names."""
        from google.longrunning.operations_pb2 import (
            GetOperationRequest,  # type: ignore
        )

        operations = []
        for name in operation_names:
            request = GetOperationRequest(name=name)
            operations.append(self.client.get_operation(request=request))
        return operations

    def is_running(self, operations: List[Operation]) -> bool:
        for op in operations:
            if not op.done:  # type: ignore[truthy-function]
                return False
        return True

    def docai_parse(self, blobs: List[Blob], gcs_output_path: str) -> List[Operation]:
        """Runs Google DocAI PDF parser on a list of blobs.

        Args:
            blobs: a list of blobs to be parsed
            gcs_output_path: a path (folder) on GCS to store results

        DocAI has a limit on the amount of documents per batch, that's why split a
            batch into mini-batches. Parsing is an async long-running operation
            on Google Cloud and results are stored in a output GCS bucket.
        """
        i = 0
        operations = []
        while i < len(blobs):
            op = self._docai_parse(
                blobs[i : (i + 4000)], gcs_output_path=gcs_output_path
            )
            operations.append(op)
            i += 4000
        return operations

    def get_results(self, operations: List[Operation]) -> List[DocAIParsingResults]:
        from google.cloud.documentai_v1 import BatchProcessMetadata

        results = []
        for op in operations:
            metadata = BatchProcessMetadata.deserialize(op.metadata.value)
            for status in metadata.individual_process_statuses:
                source = status.input_gcs_source
                output = status.output_gcs_destination
                results.append(
                    DocAIParsingResults(source_path=source, parsed_path=output)
                )
        return results

    def _docai_parse(
        self,
        blobs: List[Blob],
        gcs_output_path: str,
        *,
        enable_native_pdf_parsing: bool = True,
    ) -> Operation:
        from google.cloud import documentai
        from google.cloud.documentai_v1.types import OcrConfig, ProcessOptions

        if not self._processor_name:
            raise ValueError("Processor name is not defined, aborting!")

        documents = []
        for blob in blobs:
            gcs_document = documentai.GcsDocument(
                gcs_uri=blob.path, mime_type="application/pdf"
            )
            documents.append(gcs_document)
        gcs_documents = documentai.GcsDocuments(documents=documents)

        input_config = documentai.BatchDocumentsInputConfig(gcs_documents=gcs_documents)

        gcs_output_config = documentai.DocumentOutputConfig.GcsOutputConfig(
            gcs_uri=gcs_output_path, field_mask=None
        )
        output_config = documentai.DocumentOutputConfig(
            gcs_output_config=gcs_output_config
        )

        if enable_native_pdf_parsing:
            process_options = ProcessOptions(
                ocr_config=OcrConfig(enable_native_pdf_parsing=True)
            )
        else:
            process_options = ProcessOptions()
        request = documentai.BatchProcessRequest(
            name=self._processor_name,
            input_documents=input_config,
            document_output_config=output_config,
            process_options=process_options,
        )
        operation = self.client.batch_process_documents(request)
        return operation
