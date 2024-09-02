"""Tests for the Adobe PDF Extraction parser."""

from unittest.mock import MagicMock, patch

import pytest

from langchain_community.document_loaders import AdobePDFExtractLoader
from langchain_community.document_loaders.parsers import (
    AdobePDFExtractParser,
)


@pytest.mark.requires("adobe.pdfservices")
@patch(
    "adobe.pdfservices.operation.auth.service_principal_credentials.ServicePrincipalCredentials"
)
@patch("adobe.pdfservices.operation.pdf_services.PDFServices")
def test_adobe_pdf_services(
    mock_pdf_services: MagicMock, mock_credentials: MagicMock
) -> None:
    client_id = "client_id"
    client_secret = "client_secret"

    parser = AdobePDFExtractParser(client_id=client_id, client_secret=client_secret)
    mock_credentials.assert_called_once_with(
        client_id=client_id, client_secret=client_secret
    )
    mock_pdf_services.assert_called_once_with(credentials=mock_credentials())

    assert parser.mode == "chunks"
    assert parser.embed_figures is True
    assert isinstance(parser.credentials, mock_credentials().__class__)
    assert isinstance(parser.pdf_services, mock_pdf_services().__class__)


@pytest.mark.parametrize("mode", ["json", "chunks", "data"])
@patch(
    "adobe.pdfservices.operation.auth.service_principal_credentials.ServicePrincipalCredentials"
)
@patch("adobe.pdfservices.operation.pdf_services.PDFServices")
def test_adobe_pdf_services_loader_modes(
    mock_pdf_services: MagicMock, mock_credentials: MagicMock, mode: str
) -> None:
    loader = AdobePDFExtractLoader(
        client_id="client_id",
        client_secret="client_secret",
        file_path="path/to/file.pdf",
        mode=mode,
        embed_figures=True,
    )
    mock_credentials.assert_called_once_with(
        client_id="client_id", client_secret="client_secret"
    )
    mock_pdf_services.assert_called_once_with(credentials=mock_credentials())
    assert loader.parser.mode == mode


def test_adobe_pdf_services_loader_invalid_file_path() -> None:
    with pytest.raises(AssertionError):
        AdobePDFExtractLoader(
            client_id="client_id",
            client_secret="client_secret",
            file_path=None,  # type: ignore[arg-type]
            mode="chunks",
            embed_figures=True,
        )
