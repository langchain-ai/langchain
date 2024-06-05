import aiohttp

from langchain_community.document_loaders import SitemapLoader


class MockGet:
    def __init__(self, url, headers={},ssl=None,cookies={}):
        if "sitemap" in url:
            self._text = """
            <?xml version="1.0" encoding="UTF-8"?>
            <url>
                <loc>https://very.serious.website.org</loc>
                <lastmod>2023-05-04T16:15:31.377584+00:00</lastmod>

                <changefreq>weekly</changefreq>
                <priority>1</priority>
            </url>

            <url>
                <loc>https://example.com</loc>
                <lastmod>2023-05-05T07:52:19.633878+00:00</lastmod>

                <changefreq>daily</changefreq>
                <priority>0.9</priority>
            </url>

            </urlset>
            """
        elif "serious" in url:
            self._text = "foo"
        elif "example" in url:
            self._text = "bar"

        self.headers = headers
        self.ssl = ssl
        self.cookies = cookies

    async def text(self):
        return self._text

    async def __aexit__(self, exc_type, exc, tb):
        pass

    async def __aenter__(self):
        return self


def test__init__(mocker):
    mocker.patch.object(aiohttp.ClientSession, "get", new=MockGet)
    # session = aiohttp.ClientSession()
    loader = SitemapLoader("http://test.com/sitemap.xml", restrict_to_same_domain=False)
    docs = loader.aload()

    print(len(docs))
    assert len(docs) == 2
