import unittest
from unittest.mock import patch, Mock
from requests.exceptions import RequestException
from langchain_core.documents import Document
from langchain_community.document_loaders.steel import SteelLoader

class TestSteelLoader(unittest.TestCase):

    @patch('requests.post')
    def test_lazy_load_success_text_content(self, mock_post):
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [{"results": [{"text": "Sample text content"}]}]
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        loader = SteelLoader(api_token="fake_token", urls=["http://example.com"], text_content=True)
        documents = list(loader.lazy_load())

        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].page_content, "Sample text content")
        self.assertEqual(documents[0].metadata["source"], "http://example.com")

    @patch('requests.post')
    def test_lazy_load_success_html_content(self, mock_post):
        mock_response = Mock()
        mock_response.text = "<html>Sample HTML content</html>"
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        loader = SteelLoader(api_token="fake_token", urls=["http://example.com"], text_content=False)
        documents = list(loader.lazy_load())

        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].page_content, "<html>Sample HTML content</html>")
        self.assertEqual(documents[0].metadata["source"], "http://example.com")

    @patch('requests.post')
    def test_lazy_load_invalid_api_token(self, mock_post):
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = RequestException("Invalid API token")
        mock_post.return_value = mock_response

        loader = SteelLoader(api_token="invalid_token", urls=["http://example.com"], text_content=True)
        documents = list(loader.lazy_load())

        self.assertEqual(len(documents), 0)

    @patch('requests.post')
    def test_lazy_load_unreachable_url(self, mock_post):
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = RequestException("URL unreachable")
        mock_post.return_value = mock_response

        loader = SteelLoader(api_token="fake_token", urls=["http://unreachable-url.com"], text_content=True)
        documents = list(loader.lazy_load())

        self.assertEqual(len(documents), 0)

if __name__ == '__main__':
    unittest.main()
