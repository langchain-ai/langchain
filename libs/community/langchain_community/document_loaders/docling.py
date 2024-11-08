from enum import Enum
from typing import Any, Dict, Iterable, Iterator, Optional, Union

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document


class DoclingLoader(BaseLoader):
    """
    Docling document loader integration

    Setup:
        Install ``docling`` besides ``langchain-community``.

        .. code-block:: bash

            pip install -U docling langchain-community

    Instantiate:
        .. code-block:: python

            from langchain_community.document_loaders import DoclingLoader

            loader = DoclingLoader(
                file_path = "https://arxiv.org/pdf/2408.09869",
                # converter=...,
                # convert_kwargs=...,
                # export_type=...,
                # md_export_kwargs=...,
                # chunker=...,
            )

    Load:
        .. code-block:: python

            docs = loader.load()
            print(docs[0].page_content[:100])
            print(docs[0].metadata)

        .. code-block:: python

            ## Docling Technical Report

            Version 1.0

            Christoph Auer Maksym Lysak Ahmed Nassar Michele Dolfi Nik
            {'source': 'https://arxiv.org/pdf/2408.09869'}

    Lazy load:
        .. code-block:: python

            docs = []
            docs_lazy = loader.lazy_load()

            for doc in docs_lazy:
                docs.append(doc)
            print(docs[0].page_content[:100])
            print(docs[0].metadata)

        .. code-block:: python

            ## Docling Technical Report

            Version 1.0

            Christoph Auer Maksym Lysak Ahmed Nassar Michele Dolfi Nik
            {'source': 'https://arxiv.org/pdf/2408.09869'}
    """

    class ExportType(str, Enum):
        """Enumeration of available export types."""

        MARKDOWN = "markdown"
        DOC_CHUNKS = "doc_chunks"

    def __init__(
        self,
        file_path: Union[str, Iterable[str]],
        *,
        converter: Any = None,
        convert_kwargs: Optional[Dict[str, Any]] = None,
        export_type: ExportType = ExportType.MARKDOWN,
        md_export_kwargs: Optional[Dict[str, Any]] = None,
        chunker: Any = None,
    ):
        """Initialize with a file path.

        Args:
            file_path (Union[str, Iterable[str]]): File source as single str (URL or
                local file) or Iterable thereof.
            converter (Union[docling.document_converter.DocumentConverter, None],
                optional): Any specific `DocumentConverter` to use. Defaults to `None`
                (i.e. converter defined internally).
            convert_kwargs (Union[Dict[str, Any], None], optional): Any specific kwargs
                to pass to conversion invocation. Defaults to `None` (i.e. behavior
                defined internally).
            export_type (ExportType, optional): The type to export to: either
                `ExportType.MARKDOWN` (outputs Markdown of whole input file) or
                `ExportType.DOC_CHUNKS` (outputs chunks based on chunker). Defaults to
                `ExportType.MARKDOWN`.
            md_export_kwargs (Union[Dict[str, Any], None], optional): Any specific
                kwargs to pass to Markdown export (in case of `ExportType.MARKDOWN`).
                Defaults to `None` (i.e. behavior defined internally).
            chunker (Union[docling_core.transforms.chunker.BaseChunker, None],
                optional): Any specific `BaseChunker` to use (in case of
                `ExportType.DOC_CHUNKS`). Defaults to `None` (i.e. chunker defined
                internally).

        Raises:
            ImportError: In case `docling` is not installed.
        """

        try:
            from docling.document_converter import DocumentConverter
            from docling_core.transforms.chunker import BaseChunker, HierarchicalChunker
        except ImportError:
            raise ImportError(
                "docling package not found, please install it with `pip install docling`"  # noqa
            )

        self._file_paths = (
            file_path
            if isinstance(file_path, Iterable) and not isinstance(file_path, str)
            else [file_path]
        )

        self._converter: DocumentConverter = converter or DocumentConverter()
        self._convert_kwargs = convert_kwargs if convert_kwargs is not None else {}
        self._export_type = export_type
        self._md_export_kwargs = (
            md_export_kwargs
            if md_export_kwargs is not None
            else {"image_placeholder": ""}
        )
        self._chunker: BaseChunker = chunker or HierarchicalChunker()

    def lazy_load(
        self,
    ) -> Iterator[Document]:
        """Lazy load documents."""

        for file_path in self._file_paths:
            conv_res = self._converter.convert(
                source=file_path,
                **self._convert_kwargs,
            )
            dl_doc = conv_res.document
            if self._export_type == self.ExportType.MARKDOWN:
                yield Document(
                    page_content=dl_doc.export_to_markdown(**self._md_export_kwargs),
                    metadata={"source": file_path},
                )
            elif self._export_type == self.ExportType.DOC_CHUNKS:
                chunk_iter = self._chunker.chunk(dl_doc)
                for chunk in chunk_iter:
                    yield Document(
                        page_content=chunk.text,
                        metadata={
                            "source": file_path,
                            "dl_meta": chunk.meta.export_json_dict(),
                        },
                    )

            else:
                raise ValueError(f"Unexpected export type: {self._export_type}")
