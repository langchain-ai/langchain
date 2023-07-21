"""Loads EPub files."""
from typing import List

from langchain.document_loaders.unstructured import (
    UnstructuredFileLoader,
    satisfies_min_unstructured_version,
)


class UnstructuredEPubLoader(UnstructuredFileLoader):
    """UnstructuredEPubLoader uses unstructured to load EPUB files.
    You can run the loader in one of two modes: "single" and "elements".
    If you use "single" mode, the document will be returned as a single
    langchain Document object. If you use "elements" mode, the unstructured
    library will split the document into elements such as Title and NarrativeText.
    You can pass in additional unstructured kwargs after mode to apply
    different unstructured settings.

    Examples
    --------
    from langchain.document_loaders import UnstructuredEPubLoader

    loader = UnstructuredEPubLoader(
        "example.epub", mode="elements", strategy="fast",
    )
    docs = loader.load()

    References
    ----------
    https://unstructured-io.github.io/unstructured/bricks.html#partition-epub
    """

    def _get_elements(self) -> List:
        min_unstructured_version = "0.5.4"
        if not satisfies_min_unstructured_version(min_unstructured_version):
            raise ValueError(
                "Partitioning epub files is only supported in "
                f"unstructured>={min_unstructured_version}."
            )
        from unstructured.partition.epub import partition_epub

        return partition_epub(filename=self.file_path, **self.unstructured_kwargs)
