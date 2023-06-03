"""Load Data from a MediaWiki dump xml."""
from typing import List, Optional

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


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
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=0
            )
            texts = text_splitter.split_documents(docs)


    :param file_path: XML local file path
    :type file_path: str
    :param encoding: Charset encoding, defaults to "utf8"
    :type encoding: str, optional
    :param namespaces: The namespace of pages you want to parse. See https://www.mediawiki.org/wiki/Help:Namespaces#Localisation for a list of all common namespaces
    :type namespaces: List[int],optional
    :param skip_redirects: TR=rue to skip pages that redirect to other pages, False to keep them. False by default
    :type skip_redirects: bool, optional
    :param stop_on_error: False to skip over pages that cause parsing errors, True to stop. True by default
    :type stop_on_error: bool, optional
    """

    def __init__(
            self, file_path: str, 
            encoding: Optional[str] = "utf8",
            namespaces: Optional[List[int]] = [-1, -2, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            skip_redirects: Optional[bool]=False,
            stop_on_error: Optional[bool]=True):
        """Initialize with file path."""
        self.file_path = file_path
        self.encoding = encoding
        self.namespaces = namespaces
        self.skip_redirects = skip_redirects
        self.stop_on_error = stop_on_error

    def load(self) -> List[Document]:
        """Load from file path."""
        import mwparserfromhell
        import mwxml

        dump = mwxml.Dump.from_file(open(self.file_path, encoding=self.encoding))

        docs = []

        for page in dump.pages:
            if self.skip_redirects and page.redirect and page.redirect != None:
                continue
            if page.namespace in self.namespaces:
                try:
                    for revision in page:
                        code = mwparserfromhell.parse(revision.text)
                        text = code.strip_code(
                            normalize=True, collapse=True, keep_template_params=False
                        )
                        metadata = {"source": page.title}
                        docs.append(Document(page_content=text, metadata=metadata))
                except Exception as e:
                    print("Parsing error: {}".format(e))
                    if self.stop_on_error:
                        raise e
                    else:
                        continue
        return docs
