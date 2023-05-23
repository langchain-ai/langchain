import json
import urllib

import pytest

from langchain.document_loaders.joplin import JoplinLoader
from langchain.utils import get_from_env

try:
    access_token = get_from_env("access_token", "JOPLIN_ACCESS_TOKEN")
    url = (
        "http://localhost:41184/notes/"
        + f"?token={access_token}&fields=id,parent_id,title,body"
    )
    req_note = urllib.request.Request(url)
    with urllib.request.urlopen(req_note) as response:
        json_data = json.loads(response.read().decode())
        test_note = json_data["items"][0]

    joplin_installed = True
except:
    joplin_installed = False


@pytest.mark.skipif(not joplin_installed, reason="joplin not installed")
def test_joplin_loader() -> None:
    loader = JoplinLoader()
    docs = loader.load()

    assert type(docs) is list
    assert type(docs[0].page_content) is str
    assert type(docs[0].metadata["source"]) is str
    assert type(docs[0].metadata["title"]) is str
    assert test_note["body"] == docs[0].page_content
    assert test_note["id"] in docs[0].metadata["source"]
    assert test_note["title"] == docs[0].metadata["title"]
