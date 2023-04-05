"""Loader that loads email files."""
import os
from typing import List

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.document_loaders.unstructured import (
    UnstructuredFileLoader,
    satisfies_min_unstructured_version,
)


class UnstructuredEmailLoader(UnstructuredFileLoader):
    """Loader that uses unstructured to load email files."""

    def _get_elements(self) -> List:
        from unstructured.file_utils.filetype import FileType, detect_filetype

        filetype = detect_filetype(self.file_path)

        if filetype == FileType.EML:
            from unstructured.partition.email import partition_email

            return partition_email(filename=self.file_path)
        elif satisfies_min_unstructured_version("0.5.8") and filetype == FileType.MSG:
            from unstructured.partition.msg import partition_msg

            return partition_msg(filename=self.file_path)
        else:
            raise ValueError(
                f"Filetype {filetype} is not supported in UnstructuredEmailLoader."
            )


class OutlookMessageLoader(BaseLoader):
    """
    Loader that loads Outlook Message files using extract_msg.
    https://github.com/TeamMsgExtractor/msg-extractor
    """

    def __init__(self, file_path: str):
        """Initialize with file path."""

        self.file_path = file_path

        if not os.path.isfile(self.file_path):
            raise ValueError("File path %s is not a valid file" % self.file_path)

        try:
            import extract_msg  # noqa:F401
        except ImportError:
            raise ImportError(
                "extract_msg is not installed. Please install it with "
                "`pip install extract_msg`"
            )

    def load(self) -> List[Document]:
        """Load data into document objects."""
        import extract_msg

        msg = extract_msg.Message(self.file_path)
        return [
            Document(
                page_content=msg.body,
                metadata={
                    "subject": msg.subject,
                    "sender": msg.sender,
                    "date": msg.date,
                },
            )
        ]
