import unittest

import pytest

from langchain.document_loaders.rspace import RSpaceLoader
from pydantic.error_wrappers import ValidationError


@pytest.mark.requires("rspace_client")
class TestRSpaceLoader(unittest.TestCase):
    url = "https://community.researchspace.com"
    api_key = "myapikey"
    global_id = "SD12345"

    def test_valid_arguments(self):

        loader = RSpaceLoader(url=TestRSpaceLoader.url, api_key=TestRSpaceLoader.api_key,global_id=TestRSpaceLoader.global_id);
        self.assertEqual(TestRSpaceLoader.url, loader.url)  # add assertion here
        self.assertEqual(TestRSpaceLoader.api_key, loader.api_key)  # add assertion here
        self.assertEqual(TestRSpaceLoader.global_id, loader.global_id)  # add assertion here

    def test_missing_apikey_raises_validation_error(self):
        with self.assertRaises(ValidationError) as cm:
            RSpaceLoader(url=TestRSpaceLoader.url, global_id=TestRSpaceLoader.global_id);
        e = cm.exception
        self.assertRegex(str(e), r'Did not find api_key')

    def test_missing_url_raises_validation_error(self):
        with self.assertRaises(ValidationError) as cm:
            RSpaceLoader(api_key=TestRSpaceLoader.api_key, global_id=TestRSpaceLoader.global_id);
        e = cm.exception
        self.assertRegex(str(e), r'Did not find url')

    def test_missing_globalid_raises_validation_error(self):
        with self.assertRaises(ValidationError) as cm:
            RSpaceLoader(url=TestRSpaceLoader.url, api_key=TestRSpaceLoader.api_key);
        e = cm.exception
        self.assertRegex(str(e), r'No value supplied for global_id')

        with self.assertRaises(ValidationError) as cm:
            RSpaceLoader(url=TestRSpaceLoader.url, api_key=TestRSpaceLoader.api_key, global_id=None);
        e = cm.exception
        self.assertRegex(str(e), r'No value supplied for global_id')


if __name__ == '__main__':
    unittest.main()
