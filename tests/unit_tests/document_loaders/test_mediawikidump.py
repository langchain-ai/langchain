from pathlib import Path

import pytest

from langchain.document_loaders.mediawikidump import MWDumpLoader

PARENT_DIR = Path(__file__).parent / "sample_documents"

@pytest.mark.requires("mwparserfromhell")
def test_loading_flawed_xml()-> None:
    loader = MWDumpLoader(PARENT_DIR / "mwtest_current_pages.xml")
    try:
        loader.load()
    except Exception as e:
        assert e == ValueError

@pytest.mark.requires("mwparserfromhell")
def test_skipping_errors() -> None:
    loader = MWDumpLoader(
        file_path = PARENT_DIR / "mwtest_current_pages.xml",
        stop_on_error=False)
    documents = loader.load()
    assert len(documents) == 3

@pytest.mark.requires("mwparserfromhell")
def test_skipping_redirects() -> None:
    loader = MWDumpLoader(
        file_path = PARENT_DIR / "mwtest_current_pages.xml",
        skip_redirects=True,
        stop_on_error=False)
    documents = loader.load()
    assert len(documents) == 2

@pytest.mark.requires("mwparserfromhell")
def test_multiple_namespaces() -> None:
    loader = MWDumpLoader(
        file_path = PARENT_DIR / "mwtest_current_pages.xml",
        namespaces=[0,6],
        skip_redirects=True,
        stop_on_error=False)
    documents = loader.load()
    assert len(documents) == 3 