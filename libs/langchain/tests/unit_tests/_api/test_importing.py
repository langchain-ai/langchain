from langchain._api.module_import import create_importer


def test_import_from_non_deprecated_path() -> None:
    """Test importing all modules in langchain."""
    module_lookup = {
        "Document": "langchain_core.documents",
    }
    lookup = create_importer(__package__, module_lookup=module_lookup)
    imported_doc = lookup("Document")
    from langchain_core.documents import Document

    assert imported_doc is Document


def test_import_from_deprecated_path() -> None:
    """Test importing all modules in langchain."""
    module_lookup = {
        "Document": "langchain_core.documents",
    }
    lookup = create_importer(__package__, deprecated_lookups=module_lookup)
    imported_doc = lookup("Document")

    from langchain_core.documents import Document

    assert imported_doc is Document


def test_import_using_fallback_module() -> None:
    """Test import using fallback module."""
    lookup = create_importer(__package__, fallback_module="langchain_core.documents")
    imported_doc = lookup("Document")
    from langchain_core.documents import Document

    assert imported_doc is Document
