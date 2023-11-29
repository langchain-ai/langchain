"""Load documents from Evernote.

https://gist.github.com/foxmask/7b29c43a161e001ff04afdb2f181e31c
"""
import hashlib
import logging
from base64 import b64decode
from time import strptime
from typing import Any, Dict, Iterator, List, Optional

from langchain_core.documents import Document

from langchain.document_loaders.base import BaseLoader

logger = logging.getLogger(__name__)


class EverNoteLoader(BaseLoader):
    """Load from `EverNote`.

    Loads an EverNote notebook export file e.g. my_notebook.enex into Documents.
    Instructions on producing this file can be found at
    https://help.evernote.com/hc/en-us/articles/209005557-Export-notes-and-notebooks-as-ENEX-or-HTML

    Currently only the plain text in the note is extracted and stored as the contents
    of the Document, any non content metadata (e.g. 'author', 'created', 'updated' etc.
    but not 'content-raw' or 'resource') tags on the note will be extracted and stored
    as metadata on the Document.

    Args:
        file_path (str): The path to the notebook export with a .enex extension
        load_single_document (bool): Whether or not to concatenate the content of all
            notes into a single long Document.
        If this is set to True (default) then the only metadata on the document will be
            the 'source' which contains the file name of the export.
    """  # noqa: E501

    def __init__(self, file_path: str, load_single_document: bool = True):
        """Initialize with file path."""
        self.file_path = file_path
        self.load_single_document = load_single_document

    def load(self) -> List[Document]:
        """Load documents from EverNote export file."""
        documents = [
            Document(
                page_content=note["content"],
                metadata={
                    **{
                        key: value
                        for key, value in note.items()
                        if key not in ["content", "content-raw", "resource"]
                    },
                    **{"source": self.file_path},
                },
            )
            for note in self._parse_note_xml(self.file_path)
            if note.get("content") is not None
        ]

        if not self.load_single_document:
            return documents

        return [
            Document(
                page_content="".join([document.page_content for document in documents]),
                metadata={"source": self.file_path},
            )
        ]

    @staticmethod
    def _parse_content(content: str) -> str:
        try:
            import html2text

            return html2text.html2text(content).strip()
        except ImportError as e:
            raise ImportError(
                "Could not import `html2text`. Although it is not a required package "
                "to use Langchain, using the EverNote loader requires `html2text`. "
                "Please install `html2text` via `pip install html2text` and try again."
            ) from e

    @staticmethod
    def _parse_resource(resource: list) -> dict:
        rsc_dict: Dict[str, Any] = {}
        for elem in resource:
            if elem.tag == "data":
                # Sometimes elem.text is None
                rsc_dict[elem.tag] = b64decode(elem.text) if elem.text else b""
                rsc_dict["hash"] = hashlib.md5(rsc_dict[elem.tag]).hexdigest()
            else:
                rsc_dict[elem.tag] = elem.text

        return rsc_dict

    @staticmethod
    def _parse_note(note: List, prefix: Optional[str] = None) -> dict:
        note_dict: Dict[str, Any] = {}
        resources = []

        def add_prefix(element_tag: str) -> str:
            if prefix is None:
                return element_tag
            return f"{prefix}.{element_tag}"

        for elem in note:
            if elem.tag == "content":
                note_dict[elem.tag] = EverNoteLoader._parse_content(elem.text)
                # A copy of original content
                note_dict["content-raw"] = elem.text
            elif elem.tag == "resource":
                resources.append(EverNoteLoader._parse_resource(elem))
            elif elem.tag == "created" or elem.tag == "updated":
                note_dict[elem.tag] = strptime(elem.text, "%Y%m%dT%H%M%SZ")
            elif elem.tag == "note-attributes":
                additional_attributes = EverNoteLoader._parse_note(
                    elem, elem.tag
                )  # Recursively enter the note-attributes tag
                note_dict.update(additional_attributes)
            else:
                note_dict[elem.tag] = elem.text

        if len(resources) > 0:
            note_dict["resource"] = resources

        return {add_prefix(key): value for key, value in note_dict.items()}

    @staticmethod
    def _parse_note_xml(xml_file: str) -> Iterator[Dict[str, Any]]:
        """Parse Evernote xml."""
        # Without huge_tree set to True, parser may complain about huge text node
        # Try to recover, because there may be "&nbsp;", which will cause
        # "XMLSyntaxError: Entity 'nbsp' not defined"
        try:
            from lxml import etree
        except ImportError as e:
            logger.error(
                "Could not import `lxml`. Although it is not a required package to use "
                "Langchain, using the EverNote loader requires `lxml`. Please install "
                "`lxml` via `pip install lxml` and try again."
            )
            raise e

        context = etree.iterparse(
            xml_file, encoding="utf-8", strip_cdata=False, huge_tree=True, recover=True
        )

        for action, elem in context:
            if elem.tag == "note":
                yield EverNoteLoader._parse_note(elem)
