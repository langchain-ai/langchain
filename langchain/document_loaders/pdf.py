"""Loader that loads PDF files."""
import json
import logging
import os
import tempfile
import time
from abc import ABC
from io import StringIO
from pathlib import Path
from typing import Any, List, Optional
from urllib.parse import urlparse

import requests

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.document_loaders.unstructured import UnstructuredFileLoader
from langchain.utils import get_from_dict_or_env

logger = logging.getLogger(__file__)


class UnstructuredPDFLoader(UnstructuredFileLoader):
    """Loader that uses unstructured to load PDF files."""

    def _get_elements(self) -> List:
        from unstructured.partition.pdf import partition_pdf

        return partition_pdf(filename=self.file_path, **self.unstructured_kwargs)


class BasePDFLoader(BaseLoader, ABC):
    """Base loader class for PDF files.

    Defaults to check for local file, but if the file is a web path, it will download it
    to a temporary file, and use that, then clean up the temporary file after completion
    """

    def __init__(self, file_path: str):
        """Initialize with file path."""
        self.file_path = file_path
        self.web_path = None
        if "~" in self.file_path:
            self.file_path = os.path.expanduser(self.file_path)

        # If the file is a web path, download it to a temporary file, and use that
        if not os.path.isfile(self.file_path) and self._is_valid_url(self.file_path):
            r = requests.get(self.file_path)

            if r.status_code != 200:
                raise ValueError(
                    "Check the url of your file; returned status code %s"
                    % r.status_code
                )

            self.web_path = self.file_path
            self.temp_file = tempfile.NamedTemporaryFile()
            self.temp_file.write(r.content)
            self.file_path = self.temp_file.name
        elif not os.path.isfile(self.file_path):
            raise ValueError("File path %s is not a valid file or url" % self.file_path)

    def __del__(self) -> None:
        if hasattr(self, "temp_file"):
            self.temp_file.close()

    @staticmethod
    def _is_valid_url(url: str) -> bool:
        """Check if the url is valid."""
        parsed = urlparse(url)
        return bool(parsed.netloc) and bool(parsed.scheme)

    @property
    def source(self) -> str:
        return self.web_path if self.web_path is not None else self.file_path


class OnlinePDFLoader(BasePDFLoader):
    """Loader that loads online PDFs."""

    def load(self) -> List[Document]:
        """Load documents."""
        loader = UnstructuredPDFLoader(str(self.file_path))
        return loader.load()


class PyPDFLoader(BasePDFLoader):
    """Loads a PDF with pypdf and chunks at character level.

    Loader also stores page numbers in metadatas.
    """

    def __init__(self, file_path: str):
        """Initialize with file path."""
        try:
            import pypdf  # noqa:F401
        except ImportError:
            raise ValueError(
                "pypdf package not found, please install it with " "`pip install pypdf`"
            )
        super().__init__(file_path)

    def load(self) -> List[Document]:
        """Load given path as pages."""
        import pypdf

        with open(self.file_path, "rb") as pdf_file_obj:
            pdf_reader = pypdf.PdfReader(pdf_file_obj)
            return [
                Document(
                    page_content=page.extract_text(),
                    metadata={"source": self.file_path, "page": i},
                )
                for i, page in enumerate(pdf_reader.pages)
            ]


class PyPDFDirectoryLoader(BaseLoader):
    """Loads a directory with PDF files with pypdf and chunks at character level.

    Loader also stores page numbers in metadatas.
    """

    def __init__(
        self,
        path: str,
        glob: str = "**/[!.]*.pdf",
        silent_errors: bool = False,
        load_hidden: bool = False,
        recursive: bool = False,
    ):
        self.path = path
        self.glob = glob
        self.load_hidden = load_hidden
        self.recursive = recursive
        self.silent_errors = silent_errors

    @staticmethod
    def _is_visible(path: Path) -> bool:
        return not any(part.startswith(".") for part in path.parts)

    def load(self) -> List[Document]:
        p = Path(self.path)
        docs = []
        items = p.rglob(self.glob) if self.recursive else p.glob(self.glob)
        for i in items:
            if i.is_file():
                if self._is_visible(i.relative_to(p)) or self.load_hidden:
                    try:
                        loader = PyPDFLoader(str(i))
                        sub_docs = loader.load()
                        for doc in sub_docs:
                            doc.metadata["source"] = str(i)
                        docs.extend(sub_docs)
                    except Exception as e:
                        if self.silent_errors:
                            logger.warning(e)
                        else:
                            raise e
        return docs


class PDFMinerLoader(BasePDFLoader):
    """Loader that uses PDFMiner to load PDF files."""

    def __init__(self, file_path: str):
        """Initialize with file path."""
        try:
            from pdfminer.high_level import extract_text  # noqa:F401
        except ImportError:
            raise ValueError(
                "pdfminer package not found, please install it with "
                "`pip install pdfminer.six`"
            )

        super().__init__(file_path)

    def load(self) -> List[Document]:
        """Load file."""
        from pdfminer.high_level import extract_text

        text = extract_text(self.file_path)
        metadata = {"source": self.file_path}
        return [Document(page_content=text, metadata=metadata)]


class PDFMinerPDFasHTMLLoader(BasePDFLoader):
    """Loader that uses PDFMiner to load PDF files as HTML content."""

    def __init__(self, file_path: str):
        """Initialize with file path."""
        try:
            from pdfminer.high_level import extract_text_to_fp  # noqa:F401
        except ImportError:
            raise ValueError(
                "pdfminer package not found, please install it with "
                "`pip install pdfminer.six`"
            )

        super().__init__(file_path)

    def load(self) -> List[Document]:
        """Load file."""
        from pdfminer.high_level import extract_text_to_fp
        from pdfminer.layout import LAParams
        from pdfminer.utils import open_filename

        output_string = StringIO()
        with open_filename(self.file_path, "rb") as fp:
            extract_text_to_fp(
                fp,  # type: ignore[arg-type]
                output_string,
                codec="",
                laparams=LAParams(),
                output_type="html",
            )
        metadata = {"source": self.file_path}
        return [Document(page_content=output_string.getvalue(), metadata=metadata)]


class PyMuPDFLoader(BasePDFLoader):
    """Loader that uses PyMuPDF to load PDF files."""

    def __init__(self, file_path: str):
        """Initialize with file path."""
        try:
            import fitz  # noqa:F401
        except ImportError:
            raise ValueError(
                "PyMuPDF package not found, please install it with "
                "`pip install pymupdf`"
            )

        super().__init__(file_path)

    def load(self, **kwargs: Optional[Any]) -> List[Document]:
        """Load file."""
        import fitz

        doc = fitz.open(self.file_path)  # open document
        file_path = self.file_path if self.web_path is None else self.web_path

        return [
            Document(
                page_content=page.get_text(**kwargs).encode("utf-8"),
                metadata=dict(
                    {
                        "source": file_path,
                        "file_path": file_path,
                        "page_number": page.number + 1,
                        "total_pages": len(doc),
                    },
                    **{
                        k: doc.metadata[k]
                        for k in doc.metadata
                        if type(doc.metadata[k]) in [str, int]
                    },
                ),
            )
            for page in doc
        ]


# MathpixPDFLoader implementation taken largely from Daniel Gross's:
# https://gist.github.com/danielgross/3ab4104e14faccc12b49200843adab21
class MathpixPDFLoader(BasePDFLoader):
    def __init__(
        self,
        file_path: str,
        processed_file_format: str = "mmd",
        max_wait_time_seconds: int = 500,
        should_clean_pdf: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(file_path)
        self.mathpix_api_key = get_from_dict_or_env(
            kwargs, "mathpix_api_key", "MATHPIX_API_KEY"
        )
        self.mathpix_api_id = get_from_dict_or_env(
            kwargs, "mathpix_api_id", "MATHPIX_API_ID"
        )
        self.processed_file_format = processed_file_format
        self.max_wait_time_seconds = max_wait_time_seconds
        self.should_clean_pdf = should_clean_pdf

    @property
    def headers(self) -> dict:
        return {"app_id": self.mathpix_api_id, "app_key": self.mathpix_api_key}

    @property
    def url(self) -> str:
        return "https://api.mathpix.com/v3/pdf"

    @property
    def data(self) -> dict:
        options = {"conversion_formats": {self.processed_file_format: True}}
        return {"options_json": json.dumps(options)}

    def send_pdf(self) -> str:
        with open(self.file_path, "rb") as f:
            files = {"file": f}
            response = requests.post(
                self.url, headers=self.headers, files=files, data=self.data
            )
        response_data = response.json()
        if "pdf_id" in response_data:
            pdf_id = response_data["pdf_id"]
            return pdf_id
        else:
            raise ValueError("Unable to send PDF to Mathpix.")

    def wait_for_processing(self, pdf_id: str) -> None:
        url = self.url + "/" + pdf_id
        for _ in range(0, self.max_wait_time_seconds, 5):
            response = requests.get(url, headers=self.headers)
            response_data = response.json()
            status = response_data.get("status", None)

            if status == "completed":
                return
            elif status == "error":
                raise ValueError("Unable to retrieve PDF from Mathpix")
            else:
                print(f"Status: {status}, waiting for processing to complete")
                time.sleep(5)
        raise TimeoutError

    def get_processed_pdf(self, pdf_id: str) -> str:
        self.wait_for_processing(pdf_id)
        url = f"{self.url}/{pdf_id}.{self.processed_file_format}"
        response = requests.get(url, headers=self.headers)
        return response.content.decode("utf-8")

    def clean_pdf(self, contents: str) -> str:
        contents = "\n".join(
            [line for line in contents.split("\n") if not line.startswith("![]")]
        )
        # replace \section{Title} with # Title
        contents = contents.replace("\\section{", "# ").replace("}", "")
        # replace the "\" slash that Mathpix adds to escape $, %, (, etc.
        contents = (
            contents.replace("\$", "$")
            .replace("\%", "%")
            .replace("\(", "(")
            .replace("\)", ")")
        )
        return contents

    def load(self) -> List[Document]:
        pdf_id = self.send_pdf()
        contents = self.get_processed_pdf(pdf_id)
        if self.should_clean_pdf:
            contents = self.clean_pdf(contents)
        metadata = {"source": self.source, "file_path": self.source}
        return [Document(page_content=contents, metadata=metadata)]
