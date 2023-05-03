"""Loader that loads data from JSON."""
import json
from pathlib import Path
from typing import List, Union

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


class JSONLoader(BaseLoader):
    """Loads a JSON file and references a jq schema provided to load the text into
    documents.

    Example:
        [{"text": ...}, {"text": ...}, {"text": ...}] -> schema = .[].text
        {"key": [{"text": ...}, {"text": ...}, {"text": ...}]} -> schema = .key[].text
        ["", "", ""] -> schema = .[]
    """

    def __init__(self, file_path: Union[str, Path], jq_schema: str):
        """Initialize the JSONLoader.

        Args:
            file_path (Union[str, Path]): The path to the JSON file.
            jq_schema (str): The jq schema to use to extract the text from the JSON.
        """
        try:
            import jq  # noqa:F401
        except ImportError:
            raise ValueError(
                "jq package not found, please install it with " "`pipenv install jq`"
            )

        self.file_path = Path(file_path).resolve()
        self._jq_schema = jq.compile(jq_schema)

    def load(self) -> List[Document]:
        """Load and return documents from the JSON file."""

        body_text = self._jq_schema.input(json.loads(self.file_path.read_text())).all()
        docs = []

        for i, text in enumerate(body_text, 1):
            metadata = dict(
                source=self.file_path.as_posix(),
                seq_num=i,
            )
            docs.append(Document(page_content=text, metadata=metadata))

        return docs
