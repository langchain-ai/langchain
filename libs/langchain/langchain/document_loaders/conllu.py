"""Load CoNLL-U files."""
import csv
from typing import List

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


class CoNLLULoader(BaseLoader):
    """Load CoNLL-U files."""

    def __init__(self, file_path: str):
        """Initialize with a file path."""
        self.file_path = file_path

    def load(self) -> List[Document]:
        """Load from a file path."""
        with open(self.file_path, encoding="utf8") as f:
            tsv = list(csv.reader(f, delimiter="\t"))

            # If len(line) > 1, the line is not a comment
            lines = [line for line in tsv if len(line) > 1]

        text = ""
        for i, line in enumerate(lines):
            # Do not add a space after a punctuation mark or at the end of the sentence
            if line[9] == "SpaceAfter=No" or i == len(lines) - 1:
                text += line[1]
            else:
                text += line[1] + " "

        metadata = {"source": self.file_path}
        return [Document(page_content=text, metadata=metadata)]
