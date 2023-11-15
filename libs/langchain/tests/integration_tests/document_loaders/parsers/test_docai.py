"""Test Google Cloud DocAI parser.

You need to create a processor and enable the DocAI before running this test:

https://cloud.google.com/document-ai/docs/setup
"""
import os

from langchain.document_loaders.blob_loaders import Blob
from langchain.document_loaders.parsers import DocAIParser
from langchain.schema import Document


def test_docai_parser() -> None:
    """In order to run this test, you should provide a processor name, output path
        for DocAI to store parsing results, and an input blob path to parse.

    Example:
    export BLOB_PATH=gs://...
    export GCS_OUTPUT_PATH=gs://...
    export PROCESSOR_NAME=projects/.../locations/us/processors/...
    """
    blob_path = os.environ["BLOB_PATH"]
    gcs_output_path = os.environ["GCS_OUTPUT_PATH"]
    processor_name = os.environ["PROCESSOR_NAME"]
    parser = DocAIParser(
        location="us", processor_name=processor_name, gcs_output_path=gcs_output_path
    )
    blob = Blob(path=blob_path)
    documents = list(parser.lazy_parse(blob))
    assert len(documents) > 0
    for i, doc in enumerate(documents):
        assert isinstance(doc, Document)
        assert doc.page_content
        assert doc.metadata["source"] == blob_path
        assert doc.metadata["page"] == i + 1
