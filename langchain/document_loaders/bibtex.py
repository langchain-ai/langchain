from typing import List, Optional

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.utilities.bibtex import BibtexparserWrapper


class BibtexLoader(BaseLoader):
    """Loads a bibtex file into a list of Documents.

    Each document represents one Document.
    If a PDF file is present in the `file` bibtex field,
    the original PDF format into the text.
    """

    def __init__(
        self,
        file_path: str,
        load_max_docs: Optional[int] = 100,
        load_all_available_meta: Optional[bool] = True,
    ):
        self.file_path = file_path
        self.load_max_docs = load_max_docs
        self.load_all_available_meta = load_all_available_meta

    def load(self) -> List[Document]:
        bibtex_client = BibtexparserWrapper(
            load_max_docs=self.load_max_docs,
            load_all_available_meta=self.load_all_available_meta,
        )
        docs = bibtex_client.load(self.file_path)
        return docs
