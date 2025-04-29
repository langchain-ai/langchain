import inspect
from typing import Any, Dict, Iterator, Type

from langchain.document_loaders.base import BaseBlobParser, BaseLoader
from langchain_core._api import beta
from langchain_core.documents import Document
from langchain_core.documents.base import Blob


@beta()
class DocumentLoaderAsParser(BaseBlobParser):
    """A wrapper class that adapts a document loader to function as a parser.

    This class is a work-around that adapts a document loader to function as a parser.
    It is recommended to use a proper parser, if available.

    Requires the document loader to accept a `file_path` parameter.
    """

    DocumentLoaderType: Type[BaseLoader]
    doc_loader_kwargs: Dict[str, Any]

    def __init__(self, document_loader_class: Type[BaseLoader], **kwargs: Any) -> None:
        """
        Initializes the DocumentLoaderAsParser with a specific document loader class
        and additional arguments.

        Args:
            document_loader_class (Type[BaseLoader]): The document loader class to adapt
            as a parser.
            **kwargs: Additional arguments passed to the document loader's constructor.

        Raises:
            TypeError: If the specified document loader does not accept a `file_path` parameter,
                       an exception is raised, as only loaders with this parameter can be adapted.

        Example:
            ```
            from langchain_community.document_loaders.excel import UnstructuredExcelLoader

            # Initialize parser adapter with a document loader
            excel_parser = DocumentLoaderAsParser(UnstructuredExcelLoader, mode="elements")
            ```
        """  # noqa: E501
        super().__init__()
        self.DocumentLoaderClass = document_loader_class
        self.document_loader_kwargs = kwargs

        # Ensure the document loader class has a `file_path` parameter
        init_signature = inspect.signature(document_loader_class.__init__)
        if "file_path" not in init_signature.parameters:
            raise TypeError(
                f"{document_loader_class.__name__} does not accept `file_path`."
                "Only document loaders with `file_path` parameter"
                "can be morphed into a parser."
            )

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """
        Use underlying DocumentLoader to lazily parse the blob.
        """
        doc_loader = self.DocumentLoaderClass(  # type: ignore[call-arg]
            file_path=blob.path, **self.document_loader_kwargs
        )
        for document in doc_loader.lazy_load():
            document.metadata.update(blob.metadata)
            yield document
