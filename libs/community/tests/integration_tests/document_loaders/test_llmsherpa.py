from langchain_community.document_loaders.llmsherpa import LLMSherpaFileLoader

file_path = "https://arxiv.org/pdf/2402.14207.pdf"


def test_llmsherpa_file_loader_initialization() -> None:
    loader = LLMSherpaFileLoader(
        file_path=file_path,
    )
    docs = loader.load()
    assert isinstance(loader, LLMSherpaFileLoader)
    assert hasattr(docs, "__iter__")
    assert loader.strategy == "chunks"
    assert (
        loader.url
        == "https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all&useNewIndentParser=true&applyOcr=yes"
    )
    assert len(docs) > 1


def test_apply_ocr() -> None:
    loader = LLMSherpaFileLoader(
        file_path=file_path,
        apply_ocr=True,
        new_indent_parser=False,
    )
    docs = loader.load()
    assert (
        loader.url
        == "https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all&applyOcr=yes"
    )
    assert len(docs) > 1


def test_new_indent_parser() -> None:
    loader = LLMSherpaFileLoader(
        file_path=file_path,
        apply_ocr=False,
        new_indent_parser=True,
    )
    docs = loader.load()
    assert (
        loader.url
        == "https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all&useNewIndentParser=true"
    )
    assert len(docs) > 1
