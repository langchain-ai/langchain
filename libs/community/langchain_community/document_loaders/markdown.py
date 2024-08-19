from typing import List

from langchain_community.document_loaders.unstructured import UnstructuredFileLoader


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

    def _get_elements(self) -> List:
        from unstructured.__version__ import __version__ as __unstructured_version__
        from unstructured.partition.md import partition_md

        # NOTE(MthwRobinson) - enables the loader to work when you're using pre-release
        # versions of unstructured like 0.4.17-dev1
        _unstructured_version = __unstructured_version__.split("-")[0]
        unstructured_version = tuple([int(x) for x in _unstructured_version.split(".")])

        if unstructured_version < (0, 4, 16):
            raise ValueError(
                f"You are on unstructured version {__unstructured_version__}. "
                "Partitioning markdown files is only supported in unstructured>=0.4.16."
            )

        return partition_md(filename=self.file_path, **self.unstructured_kwargs)
