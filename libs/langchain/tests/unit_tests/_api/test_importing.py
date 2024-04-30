from langchain._api.module_import import create_importer


def test_importing_module() -> None:
    """Test importing all modules in langchain."""
    module_lookup = {
        "Document": "langchain_core.documents",
    }
    lookup = create_importer(__file__, module_lookup=module_lookup)
    imported_doc = lookup("Document")
    from langchain_core.documents import Document

    assert imported_doc is Document
