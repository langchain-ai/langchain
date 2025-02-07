"""Loads rich text files."""

from pathlib import Path
from typing import Any, List, Union

from langchain_community.document_loaders.unstructured import (
    UnstructuredFileLoader,
    validate_unstructured_version,
)


class UnstructuredRTFLoader(UnstructuredFileLoader):
    """Load `RTF` files using `Unstructured`.

    You can run the loader in one of two modes: "single" and "elements".
    If you use "single" mode, the document will be returned as a single
    langchain Document object. If you use "elements" mode, the unstructured
    library will split the document into elements such as Title and NarrativeText.
    You can pass in additional unstructured kwargs after mode to apply
    different unstructured settings.

    Examples
    --------
    from langchain_community.document_loaders import UnstructuredRTFLoader

    loader = UnstructuredRTFLoader(
        "example.rtf", mode="elements", strategy="fast",
    )
    docs = loader.load()

    References
    ----------
    https://unstructured-io.github.io/unstructured/bricks.html#partition-rtf
    """

    def __init__(
        self,
        file_path: Union[str, Path],
        mode: str = "single",
        **unstructured_kwargs: Any,
    ):
        """
        Initialize with a file path.

        Args:
            file_path: The path to the file to load.
            mode: The mode to use for partitioning. See unstructured for details.
                Defaults to "single".
            **unstructured_kwargs: Additional keyword arguments to pass
                to unstructured.
        """
        file_path = str(file_path)
        validate_unstructured_version("0.5.12")
        super().__init__(file_path=file_path, mode=mode, **unstructured_kwargs)

    def _get_elements(self) -> List:
        from unstructured.partition.rtf import partition_rtf

        return partition_rtf(filename=self.file_path, **self.unstructured_kwargs)  # type: ignore[arg-type]
