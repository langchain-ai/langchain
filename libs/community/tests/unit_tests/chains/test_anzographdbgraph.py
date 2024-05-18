# As requested, it has been included "a test for the integration, preferably unit tests that do not rely on network access". \
# We provide a python unit test that assesses the only resource not dependent on network access or containerized installation: the instantiation of the graph from a local source file.

import unittest
from unittest.mock import patch
from langchain_community.graphs import AnzoGraphDBGraph

class TestAnzoGraphDBGraph(unittest.TestCase):

    @patch('rdflib.Graph.parse')
    def test_load_local_file(self, mock_parse):
        """
        Test the loading of a local RDF file into the AnzoGraphDBGraph.
        """
        rdf_file_path = "https://github.com/SPAROntologies/foaf/blob/master/docs/current/foaf.ttl"
        local_copy_path = "test.ttl"

        graph = AnzoGraphDBGraph(
            source_file=rdf_file_path,
            standard="rdf",
            local_copy=local_copy_path
        )

        mock_parse.assert_called_once_with(rdf_file_path, format='turtle')

        self.assertIsNotNone(graph.graph)
        self.assertEqual(graph.local_copy, local_copy_path)
        self.assertEqual(graph.standard, "rdf")


if __name__ == '__main__':
    unittest.main()
