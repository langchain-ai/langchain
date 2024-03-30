import os
from typing import List

from langchain_community.document_loaders.unstructured import UnstructuredFileLoader


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

    def _get_elements(self) -> List:
        from unstructured.__version__ import __version__ as __unstructured_version__
        from unstructured.file_utils.filetype import FileType, detect_filetype

        unstructured_version = tuple(
            [int(x) for x in __unstructured_version__.split(".")]
        )
        # NOTE(MthwRobinson) - magic will raise an import error if the libmagic
        # system dependency isn't installed. If it's not installed, we'll just
        # check the file extension
        try:
            import magic  # noqa: F401

            is_ppt = detect_filetype(self.file_path) == FileType.PPT
        except ImportError:
            _, extension = os.path.splitext(str(self.file_path))
            is_ppt = extension == ".ppt"

        if is_ppt and unstructured_version < (0, 4, 11):
            raise ValueError(
                f"You are on unstructured version {__unstructured_version__}. "
                "Partitioning .ppt files is only supported in unstructured>=0.4.11. "
                "Please upgrade the unstructured package and try again."
            )

        if is_ppt:
            from unstructured.partition.ppt import partition_ppt

            return partition_ppt(filename=self.file_path, **self.unstructured_kwargs)
        else:
            from unstructured.partition.pptx import partition_pptx

            return partition_pptx(filename=self.file_path, **self.unstructured_kwargs)
