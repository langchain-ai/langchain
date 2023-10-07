import unittest

from langchain.document_loaders.rspace import RSpaceLoader


class TestRSpaceLoader(unittest.TestCase):
    url = "https://community.researchspace.com"
    api_key = "myapikey"
    global_id = "SD12345"

    def test_valid_arguments(self) -> None:
        loader = RSpaceLoader(
            url=TestRSpaceLoader.url,
            api_key=TestRSpaceLoader.api_key,
            global_id=TestRSpaceLoader.global_id,
        )
        self.assertEqual(TestRSpaceLoader.url, loader.url)  # add assertion here
        self.assertEqual(TestRSpaceLoader.api_key, loader.api_key)  # add assertion here
        self.assertEqual(
            TestRSpaceLoader.global_id, loader.global_id
        )  # add assertion here

    def test_missing_apikey_raises_validation_error(self) -> None:
        with self.assertRaises(ValueError) as cm:
            RSpaceLoader(url=TestRSpaceLoader.url, global_id=TestRSpaceLoader.global_id)
        e = cm.exception
        self.assertRegex(str(e), r"Did not find api_key")

    def test_missing_url_raises_validation_error(self) -> None:
        with self.assertRaises(ValueError) as cm:
            RSpaceLoader(
                api_key=TestRSpaceLoader.api_key, global_id=TestRSpaceLoader.global_id
            )
        e = cm.exception
        self.assertRegex(str(e), r"Did not find url")
