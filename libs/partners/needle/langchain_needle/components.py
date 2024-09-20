from haystack import component, default_from_dict, default_to_dict
from haystack.utils import Secret, deserialize_secrets_inplace
from typing import Any, Dict, List, Optional
from needle.v1 import NeedleClient
from needle.v1.models import FileToAdd

@component
class NeedleDocumentStore:
    """A Haystack-compatible document store that interfaces with Needle API collections."""

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var("NEEDLE_API_KEY"),
        name: Optional[str] = None,
        collection_id: Optional[str] = None,
        url: Optional[str] = "https://needle-ai.com",
    ):
        self.client = NeedleClient(api_key.resolve_value(), url)
        if collection_id:
            self.collection_id = collection_id
        else:
            collection = self.client.collections.create(name=name)
            self.collection_id = collection.id

    @component.output_types(collection_id=str)
    def run(self, file_urls: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Run the component. If file_urls are provided, it adds files to the collection.
        Otherwise, it just returns the collection ID."""
        if file_urls:
            self.write_documents(file_urls)
        return {"collection_id": self.collection_id}

    def write_documents(self, file_urls: Dict[str, str]) -> Dict[str, str]:
        """Add files to the Needle API collection."""
        files_to_add = [FileToAdd(name=name, url=url) for name, url in file_urls.items()]
        file_statuses = self.client.collections.files.add(collection_id=self.collection_id, files=files_to_add)

        # Polling for file indexing status
        import time
        while not all(file.status == "indexed" for file in self.client.collections.files.list(self.collection_id)):
            time.sleep(5)

        return {file.name: file.id for file in file_statuses}

    def to_dict(self) -> Dict[str, Any]:
        return default_to_dict(
            self,
            api_key=self.client.config.api_key,
            collection_id=self.collection_id,
            url=self.client.config.url,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NeedleDocumentStore":
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)


@component
class NeedleRetriever:
    """A Haystack-compatible retriever that interfaces with Needle API collections."""

    def __init__(
        self,
        document_store: NeedleDocumentStore,
    ):
        self.document_store = document_store

    @component.output_types(documents=List[Dict[str, Any]])
    def run(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Retrieve documents relevant to the query using Needle API."""
        results = self.document_store.client.collections.search(
            collection_id=self.document_store.collection_id, text=text
        )
        return {"documents": results}

    def to_dict(self) -> Dict[str, Any]:
        return default_to_dict(
            self,
            document_store=self.document_store,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NeedleRetriever":
        return default_from_dict(cls, data)
