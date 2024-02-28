from typing import List

from langchain_community.document_loaders.unstructured import UnstructuredFileLoader


class UnstructuredImageLoader(UnstructuredFileLoader):
    """Load `PNG` and `JPG` files using `Unstructured`.

    You can run the loader in one of two modes: "single" and "elements".
    If you use "single" mode, the document will be returned as a single
    langchain Document object. If you use "elements" mode, the unstructured
    library will split the document into elements such as Title and NarrativeText.
    You can pass in additional unstructured kwargs after mode to apply
    different unstructured settings.

    Examples
    --------
    from langchain_community.document_loaders import UnstructuredImageLoader

    loader = UnstructuredImageLoader(
        "example.png", mode="elements", strategy="fast",
    )
    docs = loader.load()

    References
    ----------
    https://unstructured-io.github.io/unstructured/bricks.html#partition-image
    """

    def _get_elements(self) -> List:
        from unstructured.partition.image import partition_image

        return partition_image(filename=self.file_path, **self.unstructured_kwargs)
