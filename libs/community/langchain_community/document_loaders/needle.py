from typing import Iterator, List, Optional

from langchain_core.document_loaders.base import BaseLoader
from langchain_core.documents import Document
from needle.v1 import NeedleClient
from needle.v1.models import FileToAdd


class NeedleLoader(BaseLoader):
    """
    NeedleLoader is a document loader for managing documents stored in a Needle collection.

    Setup:
        Install the `needle-python` library and set your Needle API key as an environment variable.

        .. code-block:: bash

            pip install needle-python
            export NEEDLE_API_KEY="your-api-key"

    Key init args:
        - `needle_api_key` (Optional[str]): API key for authenticating with Needle.
        - `collection_id` (str): Identifier for the Needle collection to load documents from.

    Usage:
        .. code-block:: python

            from langchain_community.document_loaders.needle import NeedleLoader

            loader = NeedleLoader(
                needle_api_key="your-api-key",
                collection_id="your-collection-id"
            )

            # Load documents
            documents = loader.load()
            for doc in documents:
                print(doc.metadata)

            # Lazy load documents
            for doc in loader.lazy_load():
                print(doc.metadata)
    """

    def __init__(
        self,
        needle_api_key: Optional[str] = None,
        collection_id: Optional[str] = None,
    ) -> None:
        """
        Initializes the NeedleLoader with API key and collection ID.

        Args:
            needle_api_key (Optional[str]): API key for authenticating with Needle.
            collection_id (Optional[str]): Identifier for the Needle collection.

        Raises:
            ValueError: If the collection ID is not provided.
        """
        super().__init__()
        self.needle_api_key = needle_api_key
        self.collection_id = collection_id
        if self.needle_api_key:
            self.client = NeedleClient(api_key=self.needle_api_key)
        else:
            self.client = None
        if not self.collection_id:
            raise ValueError("Collection ID must be provided.")

    def _get_collection(self) -> None:
        """
        Ensures the Needle collection is set and the client is initialized.

        Raises:
            ValueError: If the Needle client is not initialized or
                        if the collection ID is missing.
        """
        if not self.client:
            raise ValueError(
                "NeedleClient is not initialized. Provide a valid API key."
            )
        if not self.collection_id:
            raise ValueError("Collection ID must be provided.")

    def add_files(self, files: dict) -> None:
        """
        Adds files to the Needle collection.

        Args:
            files (dict): Dictionary where keys are file names and values are file URLs.

        Raises:
            ValueError: If the collection is not properly initialized.
        """
        self._get_collection()

        files_to_add = []
        for name, url in files.items():
            files_to_add.append(FileToAdd(name=name, url=url))

        self.client.collections.files.add(
            collection_id=self.collection_id, files=files_to_add
        )

    def _fetch_documents(self) -> List[Document]:
        """
        Fetches metadata for documents from the Needle collection.

        Returns:
            List[Document]: A list of documents with metadata. Content is excluded.

        Raises:
            ValueError: If the collection is not properly initialized.
        """
        self._get_collection()

        files = self.client.collections.files.list(self.collection_id)
        docs = []
        for file in files:
            if file.status == "indexed":
                doc = Document(
                    page_content="",  # Needle doesn't provide file content fetching
                    metadata={
                        "source": file.url,
                        "title": file.name,
                        "size": file.size if hasattr(file, "size") else None,
                    },
                )
                docs.append(doc)
        return docs

    def load(self) -> List[Document]:
        """
        Loads all documents from the Needle collection.

        Returns:
            List[Document]: A list of documents from the collection.
        """
        return self._fetch_documents()

    def lazy_load(self) -> Iterator[Document]:
        """
        Lazily loads documents from the Needle collection.

        Yields:
            Iterator[Document]: An iterator over the documents.
        """
        for doc in self._fetch_documents():
            yield doc
