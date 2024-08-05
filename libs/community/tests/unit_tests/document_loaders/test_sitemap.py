import aiohttp
import requests_mock.mocker
from langchain_community.document_loaders import SitemapLoader
import requests

xml_ = """
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9" xmlns:news="http://www.google.com/schemas/sitemap-news/0.9" xmlns:xhtml="http://www.w3.org/1999/xhtml" xmlns:image="http://www.google.com/schemas/sitemap-image/1.1" xmlns:video="http://www.google.com/schemas/sitemap-video/1.1">
<url>
<loc>https://example.com</loc>
<changefreq>weekly</changefreq>
<priority>0.5</priority>
</url>
<url>
<loc>https://test.com/production</loc>
<changefreq>weekly</changefreq>
<priority>0.5</priority>
</url>
<url>
<loc>https://test.com/next</loc>
<changefreq>weekly</changefreq>
<priority>0.5</priority>
</url>
<url>
<loc>https://important.website.org</loc>
<changefreq>weekly</changefreq>
<priority>0.5</priority>
</url> 
"""

class CustomSession(requests.Session):
    def __init__(self):
        super().__init__()

    def get(self, url, **kwargs):
        # Check if the URL matches any of the hardcoded URLs
        if url == "https://example.com":
            return self._mock_response('HERE')
        elif url == "https://test.com/production":
            return self._mock_response('NOT HERE')
        elif url == "https://test.com/next":
            return self._mock_response('ALSO NOT HERE')
        else:
            return self._mock_response("MAYBE HERE")

    def _mock_response(self, text):
        response = requests.Response()
        response.status_code = 200
        response._content = text.encode('utf-8')
        return response
    

async def test__init__():
    loader = SitemapLoader("http://test.com/sitemap.xml", restrict_to_same_domain=False,session=CustomSession())
    docs = []

    session = CustomSession()
    async with aiohttp.ClientSession() as session:
        async with session.get("https://example.com") as response:
            ans = await response.text()
    assert ans == "HERE"
    with requests_mock.mocker.Mocker() as m:
        m.get("http://test.com/sitemap.xml", text=xml_)
        docs = loader.load()
    assert len(docs) == 4


def test_lazy_load():
    loader = SitemapLoader("http://test.com/sitemap.xml", restrict_to_same_domain=False)
    docs = []

    with requests_mock.mocker.Mocker() as m:
        m.get("http://test.com/sitemap.xml", text=xml_)
        for doc in loader.lazy_load():
            docs.append(doc)

    assert len(docs) == 4


async def test_alazy_load():
    loader = SitemapLoader(
        "http://test.com/sitemap.xml", restrict_to_same_domain=False
    )
    docs = []

    with requests_mock.mocker.Mocker() as m:
        m.get("http://test.com/sitemap.xml", text=xml_)
        async for doc in loader.alazy_load():
            docs.append(doc)

    assert len(docs) == 4

async def test_same_domain():
    loader = SitemapLoader("http://test.com/sitemap.xml", restrict_to_same_domain=True)
    docs = []

    with requests_mock.mocker.Mocker() as m:
        m.get("http://test.com/sitemap.xml", text=xml_)
        async for doc in loader.alazy_load():
            docs.append(doc)  

    assert len(docs) == 2

def test_regex_filter():
    loader = SitemapLoader("http://test.com/sitemap.xml", restrict_to_same_domain=False, filter_urls=['.*example.*'])
    docs = []

    with requests_mock.mocker.Mocker() as m:
        m.get("http://test.com/sitemap.xml", text=xml_)
        m.get("https://example.com", text="HERE")
        docs = loader.load()

    assert docs[0].page_content == "HERE"
    assert len(docs) == 1

def test_blocks():
    loader = SitemapLoader("http://test.com/sitemap.xml", restrict_to_same_domain=False, blocksize=2,blocknum=1)
    docs = []

    with requests_mock.mocker.Mocker() as m:
        m.get("http://test.com/sitemap.xml", text=xml_)
        #m.get("https://test.com/next", text="HERE")
        #m.get("https://important.website.org", text="NOT HERE")
        docs = loader.load()

    #assert docs[0].page_content == "HERE"
    assert docs[0].metadata['loc'] == "https://test.com/next"    
    assert len(docs) == 2