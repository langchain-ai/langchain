import os
from typing import Any, Iterator, List

from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders.unstructured import (
    UnstructuredFileLoader,
    satisfies_min_unstructured_version,
)


class UnstructuredEmailLoader(UnstructuredFileLoader):
    """Load email files using `Unstructured`.

    Works with both
    .eml and .msg files. You can process attachments in addition to the
    e-mail message itself by passing process_attachments=True into the
    constructor for the loader. By default, attachments will be processed
    with the unstructured partition function. If you already know the document
    types of the attachments, you can specify another partitioning function
    with the attachment partitioner kwarg.

    Example
    -------
    from langchain_community.document_loaders import UnstructuredEmailLoader

    loader = UnstructuredEmailLoader("example_data/fake-email.eml", mode="elements")
    loader.load()

    Example
    -------
    from langchain_community.document_loaders import UnstructuredEmailLoader

    loader = UnstructuredEmailLoader(
        "example_data/fake-email-attachment.eml",
        mode="elements",
        process_attachments=True,
    )
    loader.load()
    """

    def __init__(
        self, file_path: str, mode: str = "single", **unstructured_kwargs: Any
    ):
        process_attachments = unstructured_kwargs.get("process_attachments")
        attachment_partitioner = unstructured_kwargs.get("attachment_partitioner")

        if process_attachments and attachment_partitioner is None:
            from unstructured.partition.auto import partition

            unstructured_kwargs["attachment_partitioner"] = partition

        super().__init__(file_path=file_path, mode=mode, **unstructured_kwargs)

    def _get_elements(self) -> List:
        from unstructured.file_utils.filetype import FileType, detect_filetype

        filetype = detect_filetype(self.file_path)

        if filetype == FileType.EML:
            from unstructured.partition.email import partition_email

            return partition_email(filename=self.file_path, **self.unstructured_kwargs)
        elif satisfies_min_unstructured_version("0.5.8") and filetype == FileType.MSG:
            from unstructured.partition.msg import partition_msg

            return partition_msg(filename=self.file_path, **self.unstructured_kwargs)
        else:
            raise ValueError(
                f"Filetype {filetype} is not supported in UnstructuredEmailLoader."
            )


class OutlookMessageLoader(BaseLoader):
    """
    Loads Outlook Message files using extract_msg.

    https://github.com/TeamMsgExtractor/msg-extractor
    """

    def __init__(self, file_path: str):
        """Initialize with a file path.

        Args:
            file_path: The path to the Outlook Message file.
        """

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

    def lazy_load(self) -> Iterator[Document]:
        import extract_msg

        msg = extract_msg.Message(self.file_path)
        yield Document(
            page_content=msg.body,
            metadata={
                "source": self.file_path,
                "subject": msg.subject,
                "sender": msg.sender,
                "date": msg.date,
            },
        )
