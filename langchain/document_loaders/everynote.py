"""Load documents from Everynote.

http://www.hanxiaogang.com/writing/parsing-evernote-export-file-enex-using-python/
"""
from base64 import b64decode
import hashlib
from lxml import etree
from io import BytesIO
import os
from time import strptime

from pypandoc import convert_text
from typing import List
from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


def parse_content(content):
    text = convert_text(content, 'org', format='html')
    return text

def parse_resource(resource):
    rsc_dict = {}
    for elem in resource:
        if elem.tag == 'data':
            # Some times elem.text is None
            rsc_dict[elem.tag] = b64decode(elem.text) if elem.text else b''
            rsc_dict['hash'] = hashlib.md5(rsc_dict[elem.tag]).hexdigest()
        else:
            rsc_dict[elem.tag] = elem.text

    return rsc_dict

def parse_note(note):
    note_dict = {}
    resources = []
    for elem in note:
        if elem.tag == 'content':
            note_dict[elem.tag] = parse_content(elem.text)
            # A copy of original content
            note_dict['content-raw'] = elem.text
        elif elem.tag == 'resource':
            resources.append(parse_resource(elem))
        elif elem.tag == 'created' or elem.tag == 'updated':
            note_dict[elem.tag] = strptime(elem.text, '%Y%m%dT%H%M%SZ')
        else:
            note_dict[elem.tag] = elem.text

    note_dict['resource'] = resources

    return note_dict

def parseNoteXML(xmlFile: str):
    # Without huge_tree set to True, parser may complain about huge text node
    # Try to recover, because there may be "&nbsp;", which will cause
    # "XMLSyntaxError: Entity 'nbsp' not defined"
    context = etree.iterparse(xmlFile, encoding='utf-8', strip_cdata=False, huge_tree=True, recover=True)
    for action, elem in context:
        if elem.tag == "note":
            yield parse_note(elem)

class EverynoteLoader(BaseLoader):
    """Loader to load in Everynote files.."""

    def __init__(self, file_path: str):
        """Initialize with file path."""
        self.file_path = file_path

    def load(self) -> List[Document]:
        """Load document from everynote files"""
        results = list(parseNoteXML(self.file_path))
        text = [r["content"] for r in results]
        metadata = {"source": self.file_path}
        return [Document(page_content=text, metadata=metadata)]
