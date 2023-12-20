"""Test BigQueryVectorSearch functionality."""

import unittest
from unittest.mock import Mock

from google.cloud import bigquery
from langchain_core.documents import Document

from langchain_community.vectorstores.bigquery_vector_search import BigQueryVectorSearch
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings

TEST_PROJECT = "bigframes-dev"
TEST_DATASET = "ashleyxu"
TEST_CONTENT = "publication_number",
TEST_VECTOR = "embedding_v1",
TEST_TABLE = "test_query_base"
TEST_SINGLE_RESULT = [Document(page_content='PL-346047-A1')]
TEST_RESULT = [Document(page_content='PL-346047-A1'),
               Document(page_content='WO-03025453-B1')]

class TestBigQueryVectorSearch(unittest.TestCase):

    def setUp(self):
        # Mocking the bigquery.Client to avoid actual API calls during testing
        self.mock_bqclient = Mock(spec=bigquery.Client)
        self.mock_bqclient.query.return_value = Mock()
        self.mock_bqclient.get_table.return_value = Mock()

        # Creating an instance of Embeddings 
        self.mock_embedding = FakeEmbeddings()

        # Creating an instance of BigQueryVectorSearch with mocked dependencies
        self.vector_search = BigQueryVectorSearch(
            project_id=TEST_PROJECT,
            dataset_name=TEST_DATASET,
            table_name=TEST_TABLE,
            content_field=TEST_CONTENT,
            vector_field=TEST_VECTOR,
            embedding=FakeEmbeddings(),
            credentials=None,
        )

        # Setting the mock bigquery.Client to the instance
        self.vector_search.bqclient = self.mock_bqclient

    def test_init(self):
        # Add your initialization tests here
        self.assertEqual(self.vector_search.project_id, TEST_PROJECT)
        self.assertEqual(self.vector_search.dataset_name, TEST_DATASET)
        self.assertEqual(self.vector_search.table_name, TEST_TABLE)
        self.assertEqual(self.vector_search.content_field, TEST_CONTENT)
        self.assertEqual(self.vector_search.vector_field, TEST_VECTOR)
        self.assertEqual(self.vector_search.embedding, FakeEmbeddings)


def test_bigqueryvectorsearch_similarity_search() -> None:
    test_vb = BigQueryVectorSearch(
            project_id=TEST_PROJECT,
            dataset_name=TEST_DATASET,
            table_name=TEST_TABLE,
            content_field=TEST_CONTENT,
            vector_field=TEST_VECTOR,
            embedding=FakeEmbeddings(),
            credentials=None,
        )
    output = test_vb.similarity_search('PL-346047-A1', k=1)
    assert output == TEST_SINGLE_RESULT