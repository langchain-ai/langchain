import os
from pathlib import Path
from typing import Sequence, Union

import pytest

from langchain_community.document_loaders import (
    AmazonTextractPDFLoader,
    MathpixPDFLoader,
    PDFMinerLoader,
    PDFMinerPDFasHTMLLoader,
    PDFPlumberLoader,
    PyMuPDFLoader,
    PyPDFium2Loader,
    PyPDFLoader,
)


def test_pypdf_loader() -> None:
    """Test PDFMiner loader."""
    file_path = Path(__file__).parent.parent / "examples/hello.pdf"
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    assert len(docs) == 1
    assert len(docs[0].metadata) == 6

    file_path = Path(__file__).parent.parent / "examples/layout-parser-paper.pdf"
    loader = PyPDFLoader(file_path)

    docs = loader.load()
    assert len(docs) == 16
    assert len(docs[0].metadata) == 13

    # Verify that extraction_mode parameter works
    file_path = Path(__file__).parent.parent / "examples/layout-parser-paper.pdf"
    loader = PyPDFLoader(
        file_path,
        mode="single",
        extract_images=False,
    )
    docs = loader.load()
    assert len(docs) == 1
    assert len(docs[0].metadata) == 12

    file_path = Path(__file__).parent.parent / "examples/layout-parser-paper.pdf"
    loader = PyPDFLoader(
        file_path,
        mode="page",
        extract_images=False,
    )
    docs = loader.load()
    assert len(docs) == 16
    assert len(docs[0].metadata) == 13

    loader = PyPDFLoader(
        file_path,
        extract_images=False,
    )
    from langchain_text_splitters import CharacterTextSplitter

    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=1000,
        chunk_overlap=0,
        separator="\n",
    )
    docs = loader.load_and_split(text_splitter)
    assert len(docs) == 18
    assert len(docs[0].metadata) == 13

    # Verify that extract_images
    file_path = Path(__file__).parent.parent / "examples/layout-parser-paper.pdf"
    loader = PyPDFLoader(
        file_path,
        mode="single",
        extract_images=True,
    )
    docs = loader.load()
    assert len(docs) == 1
    assert len(docs[0].metadata) == 12


def test_pdfplumber_loader() -> None:
    """Test PDFMiner loader."""
    file_path = Path(__file__).parent.parent / "examples/hello.pdf"
    loader = PDFPlumberLoader(file_path)
    docs = loader.load()
    assert len(docs) == 1
    assert len(docs[0].metadata) == 7

    file_path = Path(__file__).parent.parent / "examples/layout-parser-paper.pdf"
    loader = PDFPlumberLoader(file_path)

    docs = loader.load()
    assert len(docs) == 16
    assert len(docs[0].metadata) == 14

    # Verify that extraction_mode parameter works
    file_path = Path(__file__).parent.parent / "examples/layout-parser-paper.pdf"
    loader = PDFPlumberLoader(
        file_path,
        mode="single",
        extract_tables="markdown",
        extract_images=False,
    )
    docs = loader.load()
    assert len(docs) == 1
    assert len(docs[0].metadata) == 13

    file_path = Path(__file__).parent.parent / "examples/layout-parser-paper.pdf"
    loader = PDFPlumberLoader(
        file_path,
        mode="page",
        extract_tables="html",
        extract_images=False,
    )
    docs = loader.load()
    assert len(docs) == 16
    assert len(docs[0].metadata) == 14

    loader = PDFPlumberLoader(
        file_path,
        extract_tables="markdown",
        extract_images=False,
    )
    from langchain_text_splitters import CharacterTextSplitter

    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=1000,
        chunk_overlap=0,
        separator="\n",
    )
    docs = loader.load_and_split(text_splitter)
    assert len(docs) == 18
    assert len(docs[0].metadata) == 14

    # Verify that extract_tables and extract_images
    file_path = Path(__file__).parent.parent / "examples/layout-parser-paper.pdf"
    loader = PDFPlumberLoader(
        file_path,
        mode="single",
        extract_tables="markdown",
        extract_images=True,
    )
    docs = loader.load()
    assert len(docs) == 1
    assert len(docs[0].metadata) == 13


def test_pdfminer_loader() -> None:
    """Test PDFMiner loader."""
    file_path = Path(__file__).parent.parent / "examples/hello.pdf"
    loader = PDFMinerLoader(file_path)
    docs = loader.load()

    assert len(docs) == 1

    file_path = Path(__file__).parent.parent / "examples/layout-parser-paper.pdf"
    loader = PDFMinerLoader(file_path)

    docs = loader.load()
    assert len(docs) == 1

    # Verify that concatenating pages parameter works
    file_path = Path(__file__).parent.parent / "examples/hello.pdf"
    loader = PDFMinerLoader(file_path, concatenate_pages=True)
    docs = loader.load()

    assert len(docs) == 1

    file_path = Path(__file__).parent.parent / "examples/layout-parser-paper.pdf"
    loader = PDFMinerLoader(file_path, concatenate_pages=False)

    docs = loader.load()
    assert len(docs) == 16


def test_pdfminer_pdf_as_html_loader() -> None:
    """Test PDFMinerPDFasHTMLLoader."""
    file_path = Path(__file__).parent.parent / "examples/hello.pdf"
    loader = PDFMinerPDFasHTMLLoader(file_path)
    docs = loader.load()

    assert len(docs) == 1

    file_path = Path(__file__).parent.parent / "examples/layout-parser-paper.pdf"
    loader = PDFMinerPDFasHTMLLoader(file_path)

    docs = loader.load()
    assert len(docs) == 1


def test_pypdfium2_loader() -> None:
    """Test PyPDFium2Loader."""
    file_path = Path(__file__).parent.parent / "examples/hello.pdf"
    loader = PyPDFium2Loader(file_path)
    docs = loader.load()

    assert len(docs) == 1

    file_path = Path(__file__).parent.parent / "examples/layout-parser-paper.pdf"
    loader = PyPDFium2Loader(file_path)

    docs = loader.load()
    assert len(docs) == 16


def test_pymupdf_loader() -> None:
    """Test PyMuPDF loader."""
    file_path = Path(__file__).parent.parent / "examples/hello.pdf"
    loader = PyMuPDFLoader(file_path)

    docs = loader.load()
    assert len(docs) == 1

    file_path = Path(__file__).parent.parent / "examples/layout-parser-paper.pdf"
    loader = PyMuPDFLoader(file_path)

    docs = loader.load()
    assert len(docs) == 16
    assert loader.web_path is None

    web_path = "https://people.sc.fsu.edu/~jpeterson/hello_world.pdf"
    loader = PyMuPDFLoader(web_path)

    docs = loader.load()
    assert loader.web_path == web_path
    assert loader.file_path != web_path
    assert len(docs) == 1


@pytest.mark.skipif(
    not os.environ.get("MATHPIX_API_KEY"), reason="Mathpix API key not found"
)
def test_mathpix_loader() -> None:
    file_path = Path(__file__).parent.parent / "examples/hello.pdf"
    loader = MathpixPDFLoader(file_path)
    docs = loader.load()

    assert len(docs) == 1

    file_path = Path(__file__).parent.parent / "examples/layout-parser-paper.pdf"
    loader = MathpixPDFLoader(file_path)

    docs = loader.load()
    assert len(docs) == 1


@pytest.mark.parametrize(
    "file_path, features, docs_length, create_client",
    [
        (
            (
                "https://amazon-textract-public-content.s3.us-east-2.amazonaws.com"
                "/langchain/alejandro_rosalez_sample_1.jpg"
            ),
            ["FORMS", "TABLES", "LAYOUT"],
            1,
            False,
        ),
        (
            (
                "https://amazon-textract-public-content.s3.us-east-2.amazonaws.com"
                "/langchain/alejandro_rosalez_sample_1.jpg"
            ),
            [],
            1,
            False,
        ),
        (
            (
                "https://amazon-textract-public-content.s3.us-east-2.amazonaws.com"
                "/langchain/alejandro_rosalez_sample_1.jpg"
            ),
            ["TABLES"],
            1,
            False,
        ),
        (
            (
                "https://amazon-textract-public-content.s3.us-east-2.amazonaws.com"
                "/langchain/alejandro_rosalez_sample_1.jpg"
            ),
            ["FORMS"],
            1,
            False,
        ),
        (
            (
                "https://amazon-textract-public-content.s3.us-east-2.amazonaws.com"
                "/langchain/alejandro_rosalez_sample_1.jpg"
            ),
            ["LAYOUT"],
            1,
            False,
        ),
        (str(Path(__file__).parent.parent / "examples/hello.pdf"), ["FORMS"], 1, False),
        (str(Path(__file__).parent.parent / "examples/hello.pdf"), [], 1, False),
        (
            "s3://amazon-textract-public-content/langchain/layout-parser-paper.pdf",
            ["FORMS", "TABLES", "LAYOUT"],
            16,
            True,
        ),
    ],
)
@pytest.mark.skip(reason="Requires AWS credentials to run")
def test_amazontextract_loader(
    file_path: str,
    features: Union[Sequence[str], None],
    docs_length: int,
    create_client: bool,
) -> None:
    if create_client:
        import boto3

        textract_client = boto3.client("textract", region_name="us-east-2")
        loader = AmazonTextractPDFLoader(
            file_path, textract_features=features, client=textract_client
        )
    else:
        loader = AmazonTextractPDFLoader(file_path, textract_features=features)
    docs = loader.load()

    assert len(docs) == docs_length


@pytest.mark.skip(reason="Requires AWS credentials to run")
def test_amazontextract_loader_failures() -> None:
    # 2-page PDF local file system
    two_page_pdf = str(
        Path(__file__).parent.parent / "examples/multi-page-forms-sample-2-page.pdf"
    )
    loader = AmazonTextractPDFLoader(two_page_pdf)
    with pytest.raises(ValueError):
        loader.load()


# TODO pypdf loader et vérifier tous les loaders PDF s'ils ont leurs tests
