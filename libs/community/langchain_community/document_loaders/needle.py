from typing import Iterator, List, Optional
from needle.v1 import NeedleClient
from needle.v1.models import FileToAdd
from langchain_core.document_loaders.base import BaseLoader
from langchain_core.documents import Document


class NeedleLoader(BaseLoader):
    """NeedleLoader.

    Loads files from a Needle collection and converts them into LangChain `Document` objects.
    """

    needle_api_key: Optional[str] = None
    client: Optional[NeedleClient] = None
    collection_id: Optional[str] = None

    def __init__(
        self,
        needle_api_key: Optional[str] = None,
        collection_id: Optional[str] = None,
    ):
        super().__init__()
        self.needle_api_key = needle_api_key
        self.collection_id = collection_id
        if self.needle_api_key:
            self.client = NeedleClient(api_key=self.needle_api_key)
        if not self.collection_id:
            raise ValueError("Collection ID must be provided.")

    def _get_collection(self) -> None:
        """Ensures the collection ID is set."""
        if not self.client:
            raise ValueError("NeedleClient is not initialized. Provide a valid API key.")
        if not self.collection_id:
            raise ValueError("Collection ID must be provided.")

    def add_files(self, files: dict) -> None:
        """Add files to the Needle collection."""
        self._get_collection()

        files_to_add = []
        for name, url in files.items():
            files_to_add.append(FileToAdd(name=name, url=url))

        self.client.collections.files.add(collection_id=self.collection_id, files=files_to_add)

    def _fetch_documents(self) -> List[Document]:
        """Lists documents from the Needle collection."""
        self._get_collection()

        files = self.client.collections.files.list(self.collection_id)
        docs = []
        for file in files:
            if file.status == "indexed":
                doc = Document(
                    page_content="",  # Empty content since Needle doesn't provide file content fetching
                    metadata={
                        "source": file.url, 
                        "title": file.name,
                        "size": file.size if hasattr(file, 'size') else None,
                    }
                )
                docs.append(doc)
        return docs

    def load(self) -> List[Document]:
        """Load documents from the Needle collection."""
        return self._fetch_documents()
    
    def lazy_load(self) -> Iterator[Document]:
        """Lazy load documents."""
        for doc in self._fetch_documents():
            yield doc
