import requests_mock
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
import aiohttp


def mock_requests(loader):
    html1 = (
        '<div><a class="blah" href="/one">hullo</a></div>'
        '<div><a class="bleh" href="/two">buhbye</a></div>'
    )
    html2 = '<div><a class="first" href="../three">buhbye</a></div>'
    html3 = '<div><a class="second" href="../three">buhbye</a></div>'
    html4 = '<p>the end<p>'

    MOCK_DEFINITIONS = [
    ('http://test.com', html1),
    ('http://test.com/one', html2),
    ('http://test.com/two', html3),
    ('http://test.com/three', html4),
    ]

    with requests_mock.Mocker() as m:
        for url, html in MOCK_DEFINITIONS:
            m.get(url, text=html)
        docs = loader.load()
    return docs

class MockGet:
    def __init__(self, url):
        if "one" in url:
            self._text = '<div><a class="first" href="../three">buhbye</a></div>'
        elif "two" in url:
            self._text = '<div><a class="second" href="../three">buhbye</a></div>'
        elif "three" in url:
            self._text = '<p>the end<p>'
        else:
            self._text = (
                            '<div><a class="blah" href="/one">hullo</a></div>'
                            '<div><a class="bleh" href="/two">buhbye</a></div>'
                        )
        self.headers = {}

    async def text(self):
        return self._text

    async def __aexit__(self, exc_type, exc, tb):
        pass

    async def __aenter__(self):
        return self

def test_sync__init__():
    loader = RecursiveUrlLoader("http://test.com",max_depth=1)
    docs = mock_requests(loader)
    assert len(docs) == 1

def test_async__init__(mocker):
    mocker.patch.object(aiohttp.ClientSession, 'get', new=MockGet)
    loader = RecursiveUrlLoader("http://test.com",max_depth=1, use_async=True)
    docs = loader.load()
    assert len(docs) == 1
    
def test_sync_deduplication():
    loader = RecursiveUrlLoader("http://test.com",max_depth=3)
    docs = mock_requests(loader)
    assert len(docs) == 4

def test_async_deduplication(mocker):
    mocker.patch.object(aiohttp.ClientSession, 'get', new=MockGet)
    loader = RecursiveUrlLoader("http://test.com",max_depth=3, use_async=True)
    docs = loader.load()
    assert len(docs) == 4