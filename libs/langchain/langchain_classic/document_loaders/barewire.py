from typing import List, Iterator
import urllib.request
import json

from langchain_core.documents import Document
from langchain_core.document_loaders.base import BaseLoader

class BareWireLoader(BaseLoader):
    """Load web pages using BareWire's agentic proxy to strip bloat and bypass anti-bot."""

    def __init__(self, url: str, api_key: str):
        """Initialize with URL and BareWire API Key."""
        self.url = url
        self.api_key = api_key
        self.api_endpoint = "https://api.barewire.ai/v1/connect"

    def lazy_load(self) -> Iterator[Document]:
        """Lazy load the document from the BareWire proxy."""
        req = urllib.request.Request(
            self.api_endpoint,
            data=json.dumps({"url": self.url}).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
        )
        try:
            with urllib.request.urlopen(req) as response:
                content = response.read().decode("utf-8")
                # BareWire returns clean Markdown
                yield Document(page_content=content, metadata={"source": self.url})
        except Exception as e:
            raise RuntimeError(f"Error loading via BareWire: {e}")

    def load(self) -> List[Document]:
        """Load the document synchronously."""
        return list(self.lazy_load())
