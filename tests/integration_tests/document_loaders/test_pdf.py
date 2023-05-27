from pathlib import Path

from langchain.document_loaders import (
    MathpixPDFLoader,
    PDFMinerLoader,
    PDFMinerPDFasHTMLLoader,
    PyMuPDFLoader,
    PyPDFium2Loader,
    PyPDFLoader,
    UnstructuredPDFLoader,
)


def test_unstructured_pdf_loader() -> None:
    """Test unstructured loader."""
    file_path = Path(__file__).parent.parent / "examples/hello.pdf"
    loader = UnstructuredPDFLoader(str(file_path))
    docs = loader.load()

    assert len(docs) == 1


def test_pdfminer_loader() -> None:
    """Test PDFMiner loader."""
    file_path = Path(__file__).parent.parent / "examples/hello.pdf"
    loader = PDFMinerLoader(str(file_path))
    docs = loader.load()

    assert len(docs) == 1

    file_path = Path(__file__).parent.parent / "examples/layout-parser-paper.pdf"
    loader = PDFMinerLoader(str(file_path))

    docs = loader.load()
    assert len(docs) == 1


def test_pdfminer_pdf_as_html_loader() -> None:
    """Test PDFMinerPDFasHTMLLoader."""
    file_path = Path(__file__).parent.parent / "examples/hello.pdf"
    loader = PDFMinerPDFasHTMLLoader(str(file_path))
    docs = loader.load()

    assert len(docs) == 1

    file_path = Path(__file__).parent.parent / "examples/layout-parser-paper.pdf"
    loader = PDFMinerPDFasHTMLLoader(str(file_path))

    docs = loader.load()
    assert len(docs) == 1


def test_pypdf_loader() -> None:
    """Test PyPDFLoader."""
    file_path = Path(__file__).parent.parent / "examples/hello.pdf"
    loader = PyPDFLoader(str(file_path))
    docs = loader.load()

    assert len(docs) == 1

    file_path = Path(__file__).parent.parent / "examples/layout-parser-paper.pdf"
    loader = PyPDFLoader(str(file_path))

    docs = loader.load()
    assert len(docs) == 16


def test_pypdfium2_loader() -> None:
    """Test PyPDFium2Loader."""
    file_path = Path(__file__).parent.parent / "examples/hello.pdf"
    loader = PyPDFium2Loader(str(file_path))
    docs = loader.load()

    assert len(docs) == 1

    file_path = Path(__file__).parent.parent / "examples/layout-parser-paper.pdf"
    loader = PyPDFium2Loader(str(file_path))

    docs = loader.load()
    assert len(docs) == 16


def test_pymupdf_loader() -> None:
    """Test PyMuPDF loader."""
    file_path = Path(__file__).parent.parent / "examples/hello.pdf"
    loader = PyMuPDFLoader(str(file_path))

    docs = loader.load()
    assert len(docs) == 1

    file_path = Path(__file__).parent.parent / "examples/layout-parser-paper.pdf"
    loader = PyMuPDFLoader(str(file_path))

    docs = loader.load()
    assert len(docs) == 16
    assert loader.web_path is None

    web_path = "https://people.sc.fsu.edu/~jpeterson/hello_world.pdf"
    loader = PyMuPDFLoader(web_path)

    docs = loader.load()
    assert loader.web_path == web_path
    assert loader.file_path != web_path
    assert len(docs) == 1


def test_mathpix_loader() -> None:
    file_path = Path(__file__).parent.parent / "examples/hello.pdf"
    loader = MathpixPDFLoader(str(file_path))
    docs = loader.load()

    assert len(docs) == 1
    print(docs[0].page_content)

    file_path = Path(__file__).parent.parent / "examples/layout-parser-paper.pdf"
    loader = MathpixPDFLoader(str(file_path))

    docs = loader.load()
    assert len(docs) == 1
    print(docs[0].page_content)
