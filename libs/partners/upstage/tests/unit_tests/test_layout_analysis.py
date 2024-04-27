from pathlib import Path
from typing import Any, Dict, get_args
from unittest.mock import MagicMock, Mock, patch

from langchain_upstage import UpstageLayoutAnalysisLoader
from langchain_upstage.layout_analysis import OutputType, SplitType

MOCK_RESPONSE_JSON: Dict[str, Any] = {
    "api": "1.0",
    "billed_pages": 1,
    "elements": [
        {
            "bounding_box": [
                {"x": 74, "y": 906},
                {"x": 148, "y": 906},
                {"x": 148, "y": 2338},
                {"x": 74, "y": 2338},
            ],
            "category": "header",
            "html": "2021arXiv:2103.15348v2",
            "id": 0,
            "page": 1,
            "text": "arXiv:2103.15348v2",
        },
        {
            "bounding_box": [
                {"x": 654, "y": 474},
                {"x": 1912, "y": 474},
                {"x": 1912, "y": 614},
                {"x": 654, "y": 614},
            ],
            "category": "paragraph",
            "html": "LayoutParser Toolkit",
            "id": 1,
            "page": 1,
            "text": "LayoutParser Toolkit",
        },
    ],
    "html": "<header id='0'>arXiv:2103.15348v2</header>"
    + "<p id='1'>LayoutParser Toolkit</p>",
    "mimetype": "multipart/form-data",
    "model": "layout-analyzer-0.1.0",
    "text": "arXiv:2103.15348v2LayoutParser Toolkit",
}

EXAMPLE_PDF_PATH = Path(__file__).parent.parent / "examples/solar.pdf"


def test_initialization() -> None:
    """Test layout analysis document loader initialization."""
    UpstageLayoutAnalysisLoader(file_path=EXAMPLE_PDF_PATH, api_key="bar")


def test_layout_analysis_param() -> None:
    for output_type in get_args(OutputType):
        for split in get_args(SplitType):
            loader = UpstageLayoutAnalysisLoader(
                file_path=EXAMPLE_PDF_PATH,
                api_key="bar",
                output_type=output_type,
                split=split,
            )
            assert loader.output_type == output_type
            assert loader.split == split
            assert loader.api_key == "bar"
            assert loader.file_path == EXAMPLE_PDF_PATH


@patch("requests.post")
def test_none_split_text_output(mock_post: Mock) -> None:
    mock_post.return_value = MagicMock(
        status_code=200, json=MagicMock(return_value=MOCK_RESPONSE_JSON)
    )

    loader = UpstageLayoutAnalysisLoader(
        file_path=EXAMPLE_PDF_PATH,
        output_type="text",
        split="none",
        api_key="valid_api_key",
    )
    documents = loader.load()

    assert len(documents) == 1
    assert documents[0].page_content == MOCK_RESPONSE_JSON["text"]
    assert documents[0].metadata["total_pages"] == 1
    assert documents[0].metadata["type"] == "text"
    assert documents[0].metadata["split"] == "none"


@patch("requests.post")
def test_element_split_text_output(mock_post: Mock) -> None:
    mock_post.return_value = MagicMock(
        status_code=200, json=MagicMock(return_value=MOCK_RESPONSE_JSON)
    )

    loader = UpstageLayoutAnalysisLoader(
        file_path=EXAMPLE_PDF_PATH,
        output_type="text",
        split="element",
        api_key="valid_api_key",
    )
    documents = loader.load()

    assert len(documents) == 2

    for i, document in enumerate(documents):
        assert document.page_content == MOCK_RESPONSE_JSON["elements"][i]["text"]
        assert document.metadata["page"] == MOCK_RESPONSE_JSON["elements"][i]["page"]
        assert document.metadata["id"] == MOCK_RESPONSE_JSON["elements"][i]["id"]
        assert document.metadata["type"] == "text"
        assert document.metadata["split"] == "element"


@patch("requests.post")
def test_page_split_text_output(mock_post: Mock) -> None:
    mock_post.return_value = MagicMock(
        status_code=200, json=MagicMock(return_value=MOCK_RESPONSE_JSON)
    )

    loader = UpstageLayoutAnalysisLoader(
        file_path=EXAMPLE_PDF_PATH,
        output_type="text",
        split="page",
        api_key="valid_api_key",
    )
    documents = loader.load()

    assert len(documents) == 1

    for i, document in enumerate(documents):
        assert document.metadata["page"] == MOCK_RESPONSE_JSON["elements"][i]["page"]
        assert document.metadata["type"] == "text"
        assert document.metadata["split"] == "page"


@patch("requests.post")
def test_none_split_html_output(mock_post: Mock) -> None:
    mock_post.return_value = MagicMock(
        status_code=200, json=MagicMock(return_value=MOCK_RESPONSE_JSON)
    )

    loader = UpstageLayoutAnalysisLoader(
        file_path=EXAMPLE_PDF_PATH,
        output_type="html",
        split="none",
        api_key="valid_api_key",
    )
    documents = loader.load()

    assert len(documents) == 1
    assert documents[0].page_content == MOCK_RESPONSE_JSON["html"]
    assert documents[0].metadata["total_pages"] == 1
    assert documents[0].metadata["type"] == "html"
    assert documents[0].metadata["split"] == "none"


@patch("requests.post")
def test_element_split_html_output(mock_post: Mock) -> None:
    mock_post.return_value = MagicMock(
        status_code=200, json=MagicMock(return_value=MOCK_RESPONSE_JSON)
    )

    loader = UpstageLayoutAnalysisLoader(
        file_path=EXAMPLE_PDF_PATH,
        output_type="html",
        split="element",
        api_key="valid_api_key",
    )
    documents = loader.load()

    assert len(documents) == 2

    for i, document in enumerate(documents):
        assert document.page_content == MOCK_RESPONSE_JSON["elements"][i]["html"]
        assert document.metadata["page"] == MOCK_RESPONSE_JSON["elements"][i]["page"]
        assert document.metadata["id"] == MOCK_RESPONSE_JSON["elements"][i]["id"]
        assert document.metadata["type"] == "html"
        assert document.metadata["split"] == "element"


@patch("requests.post")
def test_page_split_html_output(mock_post: Mock) -> None:
    mock_post.return_value = MagicMock(
        status_code=200, json=MagicMock(return_value=MOCK_RESPONSE_JSON)
    )

    loader = UpstageLayoutAnalysisLoader(
        file_path=EXAMPLE_PDF_PATH,
        output_type="html",
        split="page",
        api_key="valid_api_key",
    )
    documents = loader.load()

    assert len(documents) == 1

    for i, document in enumerate(documents):
        assert document.metadata["page"] == MOCK_RESPONSE_JSON["elements"][i]["page"]
        assert document.metadata["type"] == "html"
        assert document.metadata["split"] == "page"
