import html
import json
import os
from abc import ABC, abstractmethod
from typing import (
    Dict,
    Iterator,
    Optional,
    Tuple,
    Union,
)

from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader


class DedocBaseLoader(BaseLoader, ABC):
    """
    Base Loader that uses `dedoc` (https://dedoc.readthedocs.io).

    Loader enables extracting text, tables and attached files from the given file:
        * `Text` can be split by pages, `dedoc` tree nodes, textual lines
            (according to the `split` parameter).
        * `Attached files` (when with_attachments=True)
            are split according to the `split` parameter.
            For attachments, langchain Document object has an additional metadata field
            `type`="attachment".
        * `Tables` (when with_tables=True) are not split - each table corresponds to one
            langchain Document object.
            For tables, Document object has additional metadata fields `type`="table"
            and `text_as_html` with table HTML representation.
    """

    def __init__(
        self,
        file_path: str,
        *,
        split: str = "document",
        with_tables: bool = True,
        with_attachments: Union[str, bool] = False,
        recursion_deep_attachments: int = 10,
        pdf_with_text_layer: str = "auto_tabby",
        language: str = "rus+eng",
        pages: str = ":",
        is_one_column_document: str = "auto",
        document_orientation: str = "auto",
        need_header_footer_analysis: Union[str, bool] = False,
        need_binarization: Union[str, bool] = False,
        need_pdf_table_analysis: Union[str, bool] = True,
        delimiter: Optional[str] = None,
        encoding: Optional[str] = None,
    ) -> None:
        """
        Initialize with file path and parsing parameters.

        Args:
            file_path: path to the file for processing
            split: type of document splitting into parts (each part is returned
                separately), default value "document"
                "document": document text is returned as a single langchain Document
                    object (don't split)
                "page": split document text into pages (works for PDF, DJVU, PPTX, PPT,
                    ODP)
                "node": split document text into tree nodes (title nodes, list item
                    nodes, raw text nodes)
                "line": split document text into lines
            with_tables: add tables to the result - each table is returned as a single
                langchain Document object

            Parameters used for document parsing via `dedoc`
                (https://dedoc.readthedocs.io/en/latest/parameters/parameters.html):

                with_attachments: enable attached files extraction
                recursion_deep_attachments: recursion level for attached files
                    extraction, works only when with_attachments==True
                pdf_with_text_layer: type of handler for parsing PDF documents,
                    available options
                    ["true", "false", "tabby", "auto", "auto_tabby" (default)]
                language: language of the document for PDF without a textual layer and
                    images, available options ["eng", "rus", "rus+eng" (default)],
                    the list of languages can be extended, please see
                    https://dedoc.readthedocs.io/en/latest/tutorials/add_new_language.html
                pages: page slice to define the reading range for parsing PDF documents
                is_one_column_document: detect number of columns for PDF without
                    a textual layer and images, available options
                    ["true", "false", "auto" (default)]
                document_orientation: fix document orientation (90, 180, 270 degrees)
                    for PDF without a textual layer and images, available options
                    ["auto" (default), "no_change"]
                need_header_footer_analysis: remove headers and footers from the output
                    result for parsing PDF and images
                need_binarization: clean pages background (binarize) for PDF without a
                    textual layer and images
                need_pdf_table_analysis: parse tables for PDF without a textual layer
                    and images
                delimiter: column separator for CSV, TSV files
                encoding: encoding of TXT, CSV, TSV
        """
        self.parsing_parameters = {
            key: value
            for key, value in locals().items()
            if key not in {"self", "file_path", "split", "with_tables"}
        }
        self.valid_split_values = {"document", "page", "node", "line"}
        if split not in self.valid_split_values:
            raise ValueError(
                f"Got {split} for `split`, but should be one of "
                f"`{self.valid_split_values}`"
            )
        self.split = split
        self.with_tables = with_tables
        self.file_path = file_path

        structure_type = "tree" if self.split == "node" else "linear"
        self.parsing_parameters["structure_type"] = structure_type
        self.parsing_parameters["need_content_analysis"] = with_attachments

    def lazy_load(self) -> Iterator[Document]:
        """Lazily load documents."""
        import tempfile

        try:
            from dedoc import DedocManager
        except ImportError:
            raise ImportError(
                "`dedoc` package not found, please install it with `pip install dedoc`"
            )
        dedoc_manager = DedocManager(manager_config=self._make_config())
        dedoc_manager.config["logger"].disabled = True

        with tempfile.TemporaryDirectory() as tmpdir:
            document_tree = dedoc_manager.parse(
                file_path=self.file_path,
                parameters={**self.parsing_parameters, "attachments_dir": tmpdir},
            )
        yield from self._split_document(
            document_tree=document_tree.to_api_schema().dict(), split=self.split
        )

    @abstractmethod
    def _make_config(self) -> dict:
        """
        Make configuration for DedocManager according to the file extension and
        parsing parameters.
        """
        pass

    def _json2txt(self, paragraph: dict) -> str:
        """Get text (recursively) of the document tree node."""
        subparagraphs_text = "\n".join(
            [
                self._json2txt(subparagraph)
                for subparagraph in paragraph["subparagraphs"]
            ]
        )
        text = (
            f"{paragraph['text']}\n{subparagraphs_text}"
            if subparagraphs_text
            else paragraph["text"]
        )
        return text

    def _parse_subparagraphs(
        self, document_tree: dict, document_metadata: dict
    ) -> Iterator[Document]:
        """Parse recursively document tree obtained by `dedoc`."""
        if len(document_tree["subparagraphs"]) > 0:
            for subparagraph in document_tree["subparagraphs"]:
                yield from self._parse_subparagraphs(
                    document_tree=subparagraph, document_metadata=document_metadata
                )
        else:
            yield Document(
                page_content=document_tree["text"],
                metadata={**document_metadata, **document_tree["metadata"]},
            )

    def _split_document(
        self,
        document_tree: dict,
        split: str,
        additional_metadata: Optional[dict] = None,
    ) -> Iterator[Document]:
        """Split document into parts according to the `split` parameter."""
        document_metadata = document_tree["metadata"]
        if additional_metadata:
            document_metadata = {**document_metadata, **additional_metadata}

        if split == "document":
            text = self._json2txt(paragraph=document_tree["content"]["structure"])
            yield Document(page_content=text, metadata=document_metadata)

        elif split == "page":
            nodes = document_tree["content"]["structure"]["subparagraphs"]
            page_id = nodes[0]["metadata"]["page_id"]
            page_text = ""

            for node in nodes:
                if node["metadata"]["page_id"] == page_id:
                    page_text += self._json2txt(node)
                else:
                    yield Document(
                        page_content=page_text,
                        metadata={**document_metadata, "page_id": page_id},
                    )
                    page_id = node["metadata"]["page_id"]
                    page_text = self._json2txt(node)

            yield Document(
                page_content=page_text,
                metadata={**document_metadata, "page_id": page_id},
            )

        elif split == "line":
            for node in document_tree["content"]["structure"]["subparagraphs"]:
                line_metadata = node["metadata"]
                yield Document(
                    page_content=self._json2txt(node),
                    metadata={**document_metadata, **line_metadata},
                )

        elif split == "node":
            yield from self._parse_subparagraphs(
                document_tree=document_tree["content"]["structure"],
                document_metadata=document_metadata,
            )

        else:
            raise ValueError(
                f"Got {split} for `split`, but should be one of "
                f"`{self.valid_split_values}`"
            )

        if self.with_tables:
            for table in document_tree["content"]["tables"]:
                table_text, table_html = self._get_table(table)
                yield Document(
                    page_content=table_text,
                    metadata={
                        **table["metadata"],
                        "type": "table",
                        "text_as_html": table_html,
                    },
                )

        for attachment in document_tree["attachments"]:
            yield from self._split_document(
                document_tree=attachment,
                split=self.split,
                additional_metadata={"type": "attachment"},
            )

    def _get_table(self, table: dict) -> Tuple[str, str]:
        """Get text and HTML representation of the table."""
        table_text = ""
        for row in table["cells"]:
            for cell in row:
                table_text += " ".join(line["text"] for line in cell["lines"])
                table_text += "\t"
            table_text += "\n"

        table_html = (
            '<table border="1" style="border-collapse: collapse; width: 100%;'
            '">\n<tbody>\n'
        )
        for row in table["cells"]:
            table_html += "<tr>\n"
            for cell in row:
                cell_text = "\n".join(line["text"] for line in cell["lines"])
                cell_text = html.escape(cell_text)
                table_html += "<td"
                if cell["invisible"]:
                    table_html += ' style="display: none" '
                table_html += (
                    f' colspan="{cell["colspan"]}" rowspan='
                    f'"{cell["rowspan"]}">{cell_text}</td>\n'
                )
            table_html += "</tr>\n"
        table_html += "</tbody>\n</table>"

        return table_text, table_html


class DedocFileLoader(DedocBaseLoader):
    """
    DedocFileLoader document loader integration to load files using `dedoc`.

    The file loader automatically detects the file type (with the correct extension).
    The list of supported file types is gives at
    https://dedoc.readthedocs.io/en/latest/index.html#id1.
    Please see the documentation of DedocBaseLoader to get more details.

    Setup:
        Install ``dedoc`` package.

        .. code-block:: bash

            pip install -U dedoc

    Instantiate:
        .. code-block:: python

            from langchain_community.document_loaders import DedocFileLoader

            loader = DedocFileLoader(
                file_path="example.pdf",
                # split=...,
                # with_tables=...,
                # pdf_with_text_layer=...,
                # pages=...,
                # ...
            )

    Load:
        .. code-block:: python

            docs = loader.load()
            print(docs[0].page_content[:100])
            print(docs[0].metadata)

        .. code-block:: python

            Some text
            {
                'file_name': 'example.pdf',
                'file_type': 'application/pdf',
                # ...
            }

    Lazy load:
        .. code-block:: python

            docs = []
            docs_lazy = loader.lazy_load()

            for doc in docs_lazy:
                docs.append(doc)
            print(docs[0].page_content[:100])
            print(docs[0].metadata)

        .. code-block:: python

            Some text
            {
                'file_name': 'example.pdf',
                'file_type': 'application/pdf',
                # ...
            }
    """

    def _make_config(self) -> dict:
        from dedoc.utils.langchain import make_manager_config

        return make_manager_config(
            file_path=self.file_path,
            parsing_params=self.parsing_parameters,
            split=self.split,
        )


class DedocAPIFileLoader(DedocBaseLoader):
    """
    Load files using `dedoc` API.
    The file loader automatically detects the file type (even with the wrong extension).
    By default, the loader makes a call to the locally hosted `dedoc` API.
    More information about `dedoc` API can be found in `dedoc` documentation:
        https://dedoc.readthedocs.io/en/latest/dedoc_api_usage/api.html

    Please see the documentation of DedocBaseLoader to get more details.

    Setup:
        You don't need to install `dedoc` library for using this loader.
        Instead, the `dedoc` API needs to be run.
        You may use Docker container for this purpose.
        Please see `dedoc` documentation for more details:
            https://dedoc.readthedocs.io/en/latest/getting_started/installation.html#install-and-run-dedoc-using-docker

        .. code-block:: bash

            docker pull dedocproject/dedoc
            docker run -p 1231:1231

    Instantiate:
        .. code-block:: python

            from langchain_community.document_loaders import DedocAPIFileLoader

            loader = DedocAPIFileLoader(
                file_path="example.pdf",
                # url=...,
                # split=...,
                # with_tables=...,
                # pdf_with_text_layer=...,
                # pages=...,
                # ...
            )

    Load:
        .. code-block:: python

            docs = loader.load()
            print(docs[0].page_content[:100])
            print(docs[0].metadata)

        .. code-block:: python

            Some text
            {
                'file_name': 'example.pdf',
                'file_type': 'application/pdf',
                # ...
            }

    Lazy load:
        .. code-block:: python

            docs = []
            docs_lazy = loader.lazy_load()

            for doc in docs_lazy:
                docs.append(doc)
            print(docs[0].page_content[:100])
            print(docs[0].metadata)

        .. code-block:: python

            Some text
            {
                'file_name': 'example.pdf',
                'file_type': 'application/pdf',
                # ...
            }
    """

    def __init__(
        self,
        file_path: str,
        *,
        url: str = "http://0.0.0.0:1231",
        split: str = "document",
        with_tables: bool = True,
        with_attachments: Union[str, bool] = False,
        recursion_deep_attachments: int = 10,
        pdf_with_text_layer: str = "auto_tabby",
        language: str = "rus+eng",
        pages: str = ":",
        is_one_column_document: str = "auto",
        document_orientation: str = "auto",
        need_header_footer_analysis: Union[str, bool] = False,
        need_binarization: Union[str, bool] = False,
        need_pdf_table_analysis: Union[str, bool] = True,
        delimiter: Optional[str] = None,
        encoding: Optional[str] = None,
    ) -> None:
        """Initialize with file path, API url and parsing parameters.

        Args:
            file_path: path to the file for processing
            url: URL to call `dedoc` API
            split: type of document splitting into parts (each part is returned
                separately), default value "document"
                "document": document is returned as a single langchain Document object
                    (don't split)
                "page": split document into pages (works for PDF, DJVU, PPTX, PPT, ODP)
                "node": split document into tree nodes (title nodes, list item nodes,
                    raw text nodes)
                "line": split document into lines
            with_tables: add tables to the result - each table is returned as a single
                langchain Document object

            Parameters used for document parsing via `dedoc`
                (https://dedoc.readthedocs.io/en/latest/parameters/parameters.html):

                with_attachments: enable attached files extraction
                recursion_deep_attachments: recursion level for attached files
                    extraction, works only when with_attachments==True
                pdf_with_text_layer: type of handler for parsing PDF documents,
                    available options
                    ["true", "false", "tabby", "auto", "auto_tabby" (default)]
                language: language of the document for PDF without a textual layer and
                    images, available options ["eng", "rus", "rus+eng" (default)],
                    the list of languages can be extended, please see
                    https://dedoc.readthedocs.io/en/latest/tutorials/add_new_language.html
                pages: page slice to define the reading range for parsing PDF documents
                is_one_column_document: detect number of columns for PDF without
                    a textual layer and images, available options
                    ["true", "false", "auto" (default)]
                document_orientation: fix document orientation (90, 180, 270 degrees)
                    for PDF without a textual layer and images, available options
                    ["auto" (default), "no_change"]
                need_header_footer_analysis: remove headers and footers from the output
                    result for parsing PDF and images
                need_binarization: clean pages background (binarize) for PDF without a
                    textual layer and images
                need_pdf_table_analysis: parse tables for PDF without a textual layer
                    and images
                delimiter: column separator for CSV, TSV files
                encoding: encoding of TXT, CSV, TSV
        """
        super().__init__(
            file_path=file_path,
            split=split,
            with_tables=with_tables,
            with_attachments=with_attachments,
            recursion_deep_attachments=recursion_deep_attachments,
            pdf_with_text_layer=pdf_with_text_layer,
            language=language,
            pages=pages,
            is_one_column_document=is_one_column_document,
            document_orientation=document_orientation,
            need_header_footer_analysis=need_header_footer_analysis,
            need_binarization=need_binarization,
            need_pdf_table_analysis=need_pdf_table_analysis,
            delimiter=delimiter,
            encoding=encoding,
        )
        self.url = url
        self.parsing_parameters["return_format"] = "json"

    def lazy_load(self) -> Iterator[Document]:
        """Lazily load documents."""
        doc_tree = self._send_file(
            url=self.url, file_path=self.file_path, parameters=self.parsing_parameters
        )
        yield from self._split_document(document_tree=doc_tree, split=self.split)

    def _make_config(self) -> dict:
        return {}

    def _send_file(
        self, url: str, file_path: str, parameters: dict
    ) -> Dict[str, Union[list, dict, str]]:
        """Send POST-request to `dedoc` API and return the results"""
        import requests

        file_name = os.path.basename(file_path)
        with open(file_path, "rb") as file:
            files = {"file": (file_name, file)}
            r = requests.post(f"{url}/upload", files=files, data=parameters)

        if r.status_code != 200:
            raise ValueError(f"Error during file handling: {r.content.decode()}")

        result = json.loads(r.content.decode())
        return result
