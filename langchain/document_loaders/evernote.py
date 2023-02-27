"""Load documents from Evernote.

https://gist.github.com/foxmask/7b29c43a161e001ff04afdb2f181e31c
"""
import hashlib
from base64 import b64decode
from time import strptime
from typing import Any, Dict, List

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


def _parse_content(content: str) -> str:
    from pypandoc import convert_text

    text = convert_text(content, "org", format="html")
    return text


def _parse_resource(resource: list) -> dict:
    rsc_dict: Dict[str, Any] = {}
    for elem in resource:
        if elem.tag == "data":
            # Some times elem.text is None
            rsc_dict[elem.tag] = b64decode(elem.text) if elem.text else b""
            rsc_dict["hash"] = hashlib.md5(rsc_dict[elem.tag]).hexdigest()
        else:
            rsc_dict[elem.tag] = elem.text

    return rsc_dict


def _parse_note(note: List) -> dict:
    note_dict: Dict[str, Any] = {}
    resources = []
    for elem in note:
        if elem.tag == "content":
            note_dict[elem.tag] = _parse_content(elem.text)
            # A copy of original content
            note_dict["content-raw"] = elem.text
        elif elem.tag == "resource":
            resources.append(_parse_resource(elem))
        elif elem.tag == "created" or elem.tag == "updated":
            note_dict[elem.tag] = strptime(elem.text, "%Y%m%dT%H%M%SZ")
        else:
            note_dict[elem.tag] = elem.text

    note_dict["resource"] = resources

    return note_dict


def _parse_note_xml(xml_file: str) -> str:
    """Parse Evernote xml."""
    # Without huge_tree set to True, parser may complain about huge text node
    # Try to recover, because there may be "&nbsp;", which will cause
    # "XMLSyntaxError: Entity 'nbsp' not defined"
    from lxml import etree

    context = etree.iterparse(
        xml_file, encoding="utf-8", strip_cdata=False, huge_tree=True, recover=True
    )
    result_string = ""
    for action, elem in context:
        if elem.tag == "note":
            result_string += _parse_note(elem)["content"]
    return result_string


class EverNoteLoader(BaseLoader):
    """Loader to load in EverNote files.."""

    def __init__(self, file_path: str):
        """Initialize with file path."""
        self.file_path = file_path

    def load(self) -> List[Document]:
        """Load document from EverNote file."""
        text = _parse_note_xml(self.file_path)
        metadata = {"source": self.file_path}
        return [Document(page_content=text, metadata=metadata)]
