from typing import Any, List, Union

from langchain_community.document_loaders.unstructured import UnstructuredFileLoader


class UnstructuredHTMLLoader(UnstructuredFileLoader):
    """Load `HTML` files using `Unstructured`.

    You can run the loader in one of two modes: "single" and "elements".
    If you use "single" mode, the document will be returned as a single
    langchain Document object. If you use "elements" mode, the unstructured
    library will split the document into elements such as Title and NarrativeText.
    You can pass in additional unstructured kwargs after mode to apply
    different unstructured settings.

    Examples
    --------
    from langchain_community.document_loaders import UnstructuredHTMLLoader

    loader = UnstructuredHTMLLoader(
        "example.html", mode="elements", strategy="fast",
    )
    docs = loader.load()

    References
    ----------
    https://unstructured-io.github.io/unstructured/bricks.html#partition-html
    """

    def __init__(
        self,
        file_path: Union[str, List[str]] = None,
        **kwargs: Any,
    ):
        def _url_to_file(url: str) -> str:
            import tempfile

            import requests

            r = requests.get(url)
            fp = tempfile.NamedTemporaryFile(mode="w", delete=False)
            fp.write(r.text)
            fp.close()
            return fp.name

        if isinstance(file_path, str):
            if file_path.startswith("http"):
                file_path = _url_to_file(file_path)
        elif isinstance(file_path, List) and any(
            [file.startswith("http") for file in file_path]
        ):
            file_path = [
                _url_to_file(file) if file.startswith("http") else file
                for file in file_path
            ]

        super().__init__(file_path=file_path, **kwargs)

    def _get_elements(self) -> List:
        from unstructured.partition.html import partition_html

        return partition_html(filename=self.file_path, **self.unstructured_kwargs)
