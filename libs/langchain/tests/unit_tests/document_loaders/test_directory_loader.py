from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.loaders.text import TextLoader


def test_directory_loader_glob_multiple() -> None:
    """Verify that globbing multiple patterns in a list works correctly."""

    path_to_tests_directory = "../../"
    extensions = [".json", ".txt", ".md"]
    list_globs = [ f"**/*{ext}" for ext in extensions]
    is_recursive_glob = True

    loader = DirectoryLoader(
        path=path_to_tests_directory,
        glob=list_globs,
        recursive=is_recursive_glob,
        loader_cls=TextLoader
    )

    list_documents = loader.load()

    is_file_type_loaded = dict(keys=extensions, values=[False]*len(extensions))

    for doc in list_documents:
        path_doc = Path(doc.metadata.get("source", ""))
        ext_doc = path_doc.suffix

        if file_types_loaded[ext_doc]:
            continue
        elif ext_doc in extensions:
            file_types_loaded[ext_doc] = True

    for ext in extensions:
        assert is_file_type_loaded[ext]
