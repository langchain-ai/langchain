import os
from pathlib import Path
from typing import Any, List, Union

from langchain_community.document_loaders.unstructured import (
    UnstructuredFileLoader,
    validate_unstructured_version,
)


class UnstructuredPowerPointLoader(UnstructuredFileLoader):
    """Load `Microsoft PowerPoint` files using `Unstructured`.

    Works with both .ppt and .pptx files.
    You can run the loader in one of two modes: "single" and "elements".
    If you use "single" mode, the document will be returned as a single
    langchain Document object. If you use "elements" mode, the unstructured
    library will split the document into elements such as Title and NarrativeText.
    You can pass in additional unstructured kwargs after mode to apply
    different unstructured settings.

    Examples
    --------
    from langchain_community.document_loaders import UnstructuredPowerPointLoader

    loader = UnstructuredPowerPointLoader(
        "example.pptx", mode="elements", strategy="fast",
    )
    docs = loader.load()

    References
    ----------
    https://unstructured-io.github.io/unstructured/bricks.html#partition-pptx
    """

    def __init__(
        self,
        file_path: Union[str, Path],
        mode: str = "single",
        **unstructured_kwargs: Any,
    ):
        """

        Args:
            file_path: The path to the PowerPoint file to load.
            mode: The mode to use when loading the file. Can be one of "single",
                "multi", or "all". Default is "single".
            **unstructured_kwargs: Any kwargs to pass to the unstructured.
        """
        file_path = str(file_path)
        super().__init__(file_path=file_path, mode=mode, **unstructured_kwargs)

    def _get_elements(self) -> List:
        from unstructured.file_utils.filetype import FileType, detect_filetype

        # NOTE(MthwRobinson) - magic will raise an import error if the libmagic
        # system dependency isn't installed. If it's not installed, we'll just
        # check the file extension
        try:
            import magic  # noqa: F401

            is_ppt = detect_filetype(self.file_path) == FileType.PPT  # type: ignore[arg-type]
        except ImportError:
            _, extension = os.path.splitext(str(self.file_path))
            is_ppt = extension == ".ppt"

        if is_ppt:
            validate_unstructured_version("0.4.11")

        if is_ppt:
            from unstructured.partition.ppt import partition_ppt

            return partition_ppt(filename=self.file_path, **self.unstructured_kwargs)  # type: ignore[arg-type]
        else:
            from unstructured.partition.pptx import partition_pptx

            return partition_pptx(filename=self.file_path, **self.unstructured_kwargs)  # type: ignore[arg-type]
