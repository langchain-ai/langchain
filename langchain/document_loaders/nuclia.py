"""Loader that loads Nuclia Understanding API results."""
import json
import uuid
from typing import List

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.tools.nuclia.tool import NucliaUnderstandingAPI


class NucliaLoader(BaseLoader):
    """Loader that loads Nuclia Understanding API results."""

    def __init__(self, path: str, nuclia_tool: NucliaUnderstandingAPI):
        self.nua = nuclia_tool
        self.id = str(uuid.uuid4())
        self.nua.run(
            {"action": "push", "id": self.id, "path": path, "enable_ml": False}
        )

    def load(self) -> List[Document]:
        """Load documents."""
        data = self.nua.run(
            {"action": "pull", "id": self.id, "path": None, "enable_ml": False}
        )
        if not data:
            return []
        obj = json.loads(data)
        text = obj["extracted_text"][0]["body"]["text"]
        print(text)
        metadata = {
            "file": obj["file_extracted_data"][0],
            "metadata": obj["field_metadata"][0],
        }
        return [Document(page_content=text, metadata=metadata)]
