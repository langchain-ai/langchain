from langchain.document_loaders.base import BaseLoader
from typing import List
from langchain.docstore.document import Document
from langchain.document_loaders.unstructured import UnstructuredLoader
from pathlib import Path


class DirectoryLoader(BaseLoader):

    def __init__(self, path: str, glob:str = "**/*"):
        self.path = path
        self.glob = glob

    def load(self) -> List[Document]:
        p = Path(self.path)
        docs = []
        for i in p.glob(self.glob):
            if i.is_file():
                sub_docs = UnstructuredLoader(str(i)).load()
                docs.extend(sub_docs)
        return docs

