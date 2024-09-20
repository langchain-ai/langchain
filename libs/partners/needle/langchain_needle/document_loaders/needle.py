from typing import Iterator, List, Optional
from needle.v1 import NeedleClient
from needle.v1.models import FileToAdd
from langchain_core.document_loaders.base import BaseLoader
from langchain_core.documents import Document
import os

class NeedleLoader(BaseLoader):
    """NeedleLoader.

    Loads files from a Needle collection and converts them into LangChain `Document` objects.
    """

    needle_api_key: Optional[str] = None
    client: Optional[NeedleClient] = None
    collection_id: Optional[str] = None
    collection_name: Optional[str] = None

    def __init__(
        self,
        needle_api_key: Optional[str] = None,
        collection_name: Optional[str] = None,
        collection_id: Optional[str] = None,
    ):
        super().__init__()
        self.needle_api_key = needle_api_key
        self.collection_name = collection_name
        self.collection_id = collection_id
        if self.needle_api_key:
            self.client = NeedleClient(api_key=self.needle_api_key)

    def _get_or_create_collection(self) -> None:
        """Fetches an existing collection or creates a new one."""
        if not self.client:
            raise ValueError("NeedleClient is not initialized. Provide a valid API key.")
        
        if not self.collection_id and self.collection_name:
            collections = self.client.collections.list()
            for collection in collections:
                if collection.name == self.collection_name:
                    self.collection_id = collection.id
                    return
            # If collection doesn't exist, create it
            collection = self.client.collections.create(name=self.collection_name)
            self.collection_id = collection.id
        elif not self.collection_id and not self.collection_name:
            raise ValueError("Collection ID or name must be provided.")

    def add_files(self, files: List[dict]) -> None:
        """Add files to the Needle collection."""
        if not self.client:
            raise ValueError("NeedleClient is not initialized. Provide a valid API key.")
        
        self._get_or_create_collection()

        files_to_add = []
        for name, url in files.items():
            files_to_add.append(FileToAdd(name=name, url=url))

        file_statuses = self.client.collections.files.add(collection_id=self.collection_id, files=files_to_add)

        # Optionally, implement polling logic to wait for files to be indexed
        # Insert polling logic here if necessary

    def _fetch_documents(self) -> List[Document]:
        """Fetch documents from the Needle collection."""
        if not self.client:
            raise ValueError("NeedleClient is not initialized. Provide a valid API key.")
        
        if not self.collection_id:
            raise ValueError("Collection ID not set. Call add_files or load to set or fetch collection.")
        
        files = self.client.collections.files.list(self.collection_id)
        docs = []
        for file in files:
            if file.status == "indexed":
                content = self.client.files.get_content(file.id)  # Assuming there's a method to fetch file content
                doc = Document(page_content=content, metadata={"source": file.url, "title": file.name})
                docs.append(doc)
        return docs

    def load(self) -> List[Document]:
        """Load documents from the Needle collection."""
        if not self.client:
            raise ValueError("NeedleClient is not initialized. Provide a valid API key.")
        
        self._get_or_create_collection()
        return self._fetch_documents()
    
    def lazy_load(self) -> Iterator[Document]:
        """Lazy load documents."""
        if not self.client:
            raise ValueError("NeedleClient is not initialized. Provide a valid API key.")
        
        self._get_or_create_collection()
        for doc in self._fetch_documents():
            yield doc
