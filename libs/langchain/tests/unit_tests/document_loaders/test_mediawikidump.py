from pathlib import Path

import pytest

from langchain.document_loaders.mediawikidump import MWDumpLoader

PARENT_DIR = Path(__file__).parent / "sample_documents"


@pytest.mark.requires("mwparserfromhell", "mwxml")
def test_loading_flawed_xml() -> None:
    loader = MWDumpLoader((PARENT_DIR / "mwtest_current_pages.xml").absolute())
    with pytest.raises(TypeError):
        loader.load()


@pytest.mark.requires("mwparserfromhell", "mwxml")
def test_skipping_errors() -> None:
    loader = MWDumpLoader(
        file_path=(PARENT_DIR / "mwtest_current_pages.xml").absolute(),
        stop_on_error=False,
    )
    documents = loader.load()
    assert len(documents) == 3


@pytest.mark.requires("mwparserfromhell", "mwxml")
def test_skipping_redirects() -> None:
    loader = MWDumpLoader(
        file_path=(PARENT_DIR / "mwtest_current_pages.xml").absolute(),
        skip_redirects=True,
        stop_on_error=False,
    )
    documents = loader.load()
    assert len(documents) == 2


@pytest.mark.requires("mwparserfromhell", "mwxml")
def test_multiple_namespaces() -> None:
    loader = MWDumpLoader(
        file_path=(PARENT_DIR / "mwtest_current_pages.xml").absolute(),
        namespaces=[0, 6],
        skip_redirects=True,
        stop_on_error=False,
    )
    documents = loader.load()
    [print(doc) for doc in documents]
    assert len(documents) == 2
