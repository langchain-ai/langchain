"""Test common functions."""
import json
import unittest
from pathlib import Path

from langchain_robocorp._common import reduce_openapi_spec

from ._fixtures import openapi_endpoint_doc_mock


class TestReduceOpenAPISpec(unittest.TestCase):
    maxDiff = None

    def test_reduce_openapi_spec(self) -> None:
        with Path(__file__).with_name("_openapi.fixture.json").open("r") as file:
            original = json.load(file)

        output = reduce_openapi_spec("https://foo.bar", original)

        self.assertEqual(output.servers, [{"url": "https://foo.bar"}])
        self.assertEqual(output.description, "Robocorp Actions Server")

        self.assertEqual(len(output.endpoints), 8)
        self.assertEqual(output.endpoints[0][1], openapi_endpoint_doc_mock)
