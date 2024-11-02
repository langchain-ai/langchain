import unittest
import os
from simplesearx import SSearX

class TestSSearX(unittest.TestCase):
    def setUp(self):
        # Set the USER_AGENT environment variable for requests
        os.environ['USER_AGENT'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36'

        # Instantiate SSearX
        self.ssx = SSearX()

    def test_google_search(self):
        # Perform a Google search
        result = self.ssx.ssxGoogle("LangChain")
        
        # Check that the result is not None
        self.assertIsNotNone(result)
        
        # Verify if "LangChain" appears in the metadata or content
        if result:
            content = result[0].page_content if isinstance(result, list) else result.page_content
            self.assertIn("LangChain", content)

    def test_website_scraping(self):
        # Test a website scraping
        result = self.ssx.ssxWebsite("example.com")
        self.assertIsNotNone(result)

if __name__ == "__main__":
    unittest.main()
