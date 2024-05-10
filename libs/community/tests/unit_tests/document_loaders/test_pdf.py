import io
from unittest.mock import patch

import pytest

from langchain_community.document_loaders.pdf import (
    BasePDFLoader,
)


@pytest.fixture
def pdf_data():
    return b'%PDF-1.4\n%\xc2\xb5\xc2\xb3\n'  # Sample PDF data

def test_base_pdf_loader_with_bytesio(pdf_data):
    file_obj = io.BytesIO(pdf_data)
    loader = BasePDFLoader(file_obj=file_obj)
    assert loader.source == file_obj

def test_base_pdf_loader_with_file_path(pdf_data, tmp_path):
    file_path = tmp_path / "test.pdf"
    file_path.write_bytes(pdf_data)

    loader = BasePDFLoader(file_path=str(file_path))
    assert loader.source == str(file_path)

def test_base_pdf_loader_with_web_path(pdf_data):
    web_path = "https://example.com/test.pdf"
    with patch("requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.content = pdf_data

        loader = BasePDFLoader(file_path=web_path)
        assert loader.source.startswith(loader.temp_dir.name)

def test_base_pdf_loader_with_invalid_path():
    invalid_path = "invalid_path"
    with pytest.raises(ValueError):
        BasePDFLoader(file_path=invalid_path)