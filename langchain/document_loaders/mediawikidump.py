"""Load Data from a MediaWiki dump xml"""
import os
from typing import List, Optional
from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
import pkg_resources
import mwparserfromhell
import mwxml

dependencies = [
    'mwtypes>=0.4.0',
    'mwxml>=0.3.4'
]  
try:
    pkg_resources.require(dependencies)
except pkg_resources.VersionConflict as version_error:
    print("The following modules caused an error:")
    print("Version installed :", version_error.dist)
    print("Version required  :", version_error.req)
    print("You may pip install git+https://github.com/gdedrouas/python-mwxml@xml_format_0.11")
    raise

            
class MWDumpLoader(BaseLoader):
    """
    Load MediaWiki dump from XML file
    Example:
        .. code-block:: python

            from langchain.document_loaders import MWDumpLoader

            loader = MWDumpLoader(
                file_path="myWiki.xml",
                encoding="utf8"
            )
            docs = loader.load()
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            texts = text_splitter.split_documents(docs)
            
            
    :param file_path: XML local file path
    :type file_path: str
    :param encoding: Charset encoding, defaults to "utf8"
    :type encoding: str, optional
    """
    def __init__(self, file_path: str, encoding: Optional[str] = "utf8"):
        """Initialize with file path."""
        self.file_path = file_path
        self.encoding = encoding
        
    def load(self) -> List[Document]:
        """Load from file path."""
        dump = mwxml.Dump.from_file(open(self.file_path, encoding=self.encoding))

        docs = []
        
        for page in dump.pages:
            for revision in page:
                code = mwparserfromhell.parse(revision.text)
                text = code.strip_code(normalize=True, collapse=True, keep_template_params=False)
                metadata = {"source": page.title}
                docs.append(Document(page_content=text, metadata=metadata))
        
        return docs
