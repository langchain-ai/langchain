import logging
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Optional, Sequence, Union

from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import mwxml


class MWDumpLoader(BaseLoader):
    """Load `MediaWiki` dump from an `XML` file.

    Example:
        .. code-block:: python

            from langchain_text_splitters import RecursiveCharacterTextSplitter
            from langchain_community.document_loaders import MWDumpLoader

            loader = MWDumpLoader(
                file_path="myWiki.xml",
                encoding="utf8"
            )
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=0
            )
            texts = text_splitter.split_documents(docs)


    :param file_path: XML local file path
    :type file_path: str
    :param encoding: Charset encoding, defaults to "utf8"
    :type encoding: str, optional
    :param namespaces: The namespace of pages you want to parse.
        See https://www.mediawiki.org/wiki/Help:Namespaces#Localisation
        for a list of all common namespaces
    :type namespaces: List[int],optional
    :param skip_redirects: TR=rue to skip pages that redirect to other pages,
        False to keep them. False by default
    :type skip_redirects: bool, optional
    :param stop_on_error: False to skip over pages that cause parsing errors,
        True to stop. True by default
    :type stop_on_error: bool, optional
    """

    def __init__(
        self,
        file_path: Union[str, Path],
        encoding: Optional[str] = "utf8",
        namespaces: Optional[Sequence[int]] = None,
        skip_redirects: Optional[bool] = False,
        stop_on_error: Optional[bool] = True,
    ):
        self.file_path = file_path if isinstance(file_path, str) else str(file_path)
        self.encoding = encoding
        # Namespaces range from -2 to 15, inclusive.
        self.namespaces = namespaces
        self.skip_redirects = skip_redirects
        self.stop_on_error = stop_on_error

    def _load_dump_file(self) -> "mwxml.Dump":
        try:
            import mwxml
        except ImportError as e:
            raise ImportError(
                "Unable to import 'mwxml'. Please install with `pip install mwxml`."
            ) from e

        return mwxml.Dump.from_file(open(self.file_path, encoding=self.encoding))

    def _load_single_page_from_dump(self, page: "mwxml.Page") -> Document:  # type: ignore[return]
        """Parse a single page."""
        try:
            import mwparserfromhell
        except ImportError as e:
            raise ImportError(
                "Unable to import 'mwparserfromhell'. Please install with"
                " `pip install mwparserfromhell`."
            ) from e
        for revision in page:
            code = mwparserfromhell.parse(revision.text)
            text = code.strip_code(
                normalize=True, collapse=True, keep_template_params=False
            )
            metadata = {"source": page.title}
            return Document(page_content=text, metadata=metadata)

    def lazy_load(
        self,
    ) -> Iterator[Document]:
        """Lazy load from a file path."""

        dump = self._load_dump_file()

        for page in dump.pages:
            if self.skip_redirects and page.redirect:
                continue
            if self.namespaces and page.namespace not in self.namespaces:
                continue
            try:
                yield self._load_single_page_from_dump(page)
            except Exception as e:
                logger.error("Parsing error: {}".format(e))
                if self.stop_on_error:
                    raise e
                else:
                    continue
