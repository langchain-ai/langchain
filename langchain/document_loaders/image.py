"""Loader that loads image files."""
from typing import List

from langchain.document_loaders.unstructured import UnstructuredFileLoader


class UnstructuredImageLoader(UnstructuredFileLoader):
    """Loader that uses unstructured to load image files, such as PNGs and JPGs."""

    def _get_elements(self) -> List:
        from unstructured.partition.image import partition_image

        return partition_image(filename=self.file_path, **self.unstructured_kwargs)
