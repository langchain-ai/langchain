import unittest

import pytest


class TestOntotextGraphDBGraph(unittest.TestCase):
    def test_import(self) -> None:
        from langchain_community.graphs import OntotextGraphDBGraph  # noqa: F401

    @pytest.mark.requires("SPARQLWrapper")
    def test_check_connectivity_no_connectivity(self) -> None:
        from langchain_community.graphs import OntotextGraphDBGraph

        with self.assertRaises(ValueError) as e:
            OntotextGraphDBGraph(
                gdb_repository="http://localhost:7200/repositories/non-existing-repository"
            )
        self.assertEqual(
            "Could not query the provided repository. "
            "Please, check, if the value of the provided "
            "gdb_repository points to the right repository. "
            "If GraphDB is secured, please, make sure that the environment variables "
            "'GRAPHDB_USERNAME' and 'GRAPHDB_PASSWORD' are set, "
            "or the correct authentication headers are set in custom_http_headers.",
            str(e.exception),
        )
