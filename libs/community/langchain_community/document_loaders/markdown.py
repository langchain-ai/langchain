from pathlib import Path
from typing import Any, List, Union

from langchain_community.document_loaders.unstructured import (
    UnstructuredFileLoader,
    validate_unstructured_version,
)


class UnstructuredMarkdownLoader(UnstructuredFileLoader):
    """Load `Markdown` files using `Unstructured`.

    You can run the loader in one of two modes: "single" and "elements".
    If you use "single" mode, the document will be returned as a single
    langchain Document object. If you use "elements" mode, the unstructured
    library will split the document into elements such as Title and NarrativeText.
    You can pass in additional unstructured kwargs after mode to apply
    different unstructured settings.

    Setup:
        Install ``langchain-community``.

        .. code-block:: bash

            pip install -U langchain-community

    Instantiate:
        .. code-block:: python

            from langchain_community.document_loaders import UnstructuredMarkdownLoader

            loader = UnstructuredMarkdownLoader(
                "./example_data/example.md",
                mode="elements",
                strategy="fast",
            )

    Lazy load:
        .. code-block:: python

            docs = []
            docs_lazy = loader.lazy_load()

            # async variant:
            # docs_lazy = await loader.alazy_load()

            for doc in docs_lazy:
                docs.append(doc)
            print(docs[0].page_content[:100])
            print(docs[0].metadata)

        .. code-block:: python

            Sample Markdown Document
            {'source': './example_data/example.md', 'category_depth': 0, 'last_modified': '2024-08-14T15:04:18', 'languages': ['eng'], 'filetype': 'text/markdown', 'file_directory': './example_data', 'filename': 'example.md', 'category': 'Title', 'element_id': '3d0b313864598e704aa26c728ecb61e5'}


    Async load:
        .. code-block:: python

            docs = await loader.aload()
            print(docs[0].page_content[:100])
            print(docs[0].metadata)

        .. code-block:: python

            Sample Markdown Document
            {'source': './example_data/example.md', 'category_depth': 0, 'last_modified': '2024-08-14T15:04:18', 'languages': ['eng'], 'filetype': 'text/markdown', 'file_directory': './example_data', 'filename': 'example.md', 'category': 'Title', 'element_id': '3d0b313864598e704aa26c728ecb61e5'}

    References
    ----------
    https://unstructured-io.github.io/unstructured/core/partition.html#partition-md
    """  # noqa: E501

    def __init__(
        self,
        file_path: Union[str, Path],
        mode: str = "single",
        **unstructured_kwargs: Any,
    ):
        """

        Args:
            file_path: The path to the Markdown file to load.
            mode: The mode to use when loading the file. Can be one of "single",
                "multi", or "all". Default is "single".
            **unstructured_kwargs: Any kwargs to pass to the unstructured.
        """
        file_path = str(file_path)
        validate_unstructured_version("0.4.16")
        super().__init__(file_path=file_path, mode=mode, **unstructured_kwargs)

    def _get_elements(self) -> List:
        from unstructured.partition.md import partition_md

        return partition_md(filename=self.file_path, **self.unstructured_kwargs)
