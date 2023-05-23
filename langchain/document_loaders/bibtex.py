from typing import Iterator, List

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.utilities.bibtex import BibtexparserWrapper


class BibtexLoader(BaseLoader):
    """Loads a bibtex file into a list of Documents.

    Each document represents one entry from the bibtex file.

    If a PDF file is present in the `file` bibtex field, the original PDF
    is loaded into the document text. If no such file entry is present,
    the `abstract` field is used instead.
    """

    def __init__(
        self,
        file_path: str,
        load_max_docs: int = 100,
        load_all_available_meta: bool = True,
    ):
        """Initialize the BibtexLoader.

        Args:
            file_path: Path to the bibtex file.
            load_max_docs: Max number of associated documents to load. Use -1 means
                           no limit.
            load_all_available_meta: Load all available metadata or restrict
                                     to most informative fields.
        """
        self.file_path = file_path
        self.load_max_docs = load_max_docs
        self.load_all_available_meta = load_all_available_meta

    def load(self) -> List[Document]:
        """Load the documents."""
        return list(self.lazy_load())

    def lazy_load(
        self,
    ) -> Iterator[Document]:
        """Load documents lazily."""
        bibtex_client = BibtexparserWrapper(
            load_max_docs=self.load_max_docs,
            load_all_available_meta=self.load_all_available_meta,
        )
        yield from bibtex_client.lazy_load(self.file_path)
