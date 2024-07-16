"""Test ApertureDB functionality."""

import uuid

import pytest
from langchain_standard_tests.integration_tests.vectorstores import (
    AsyncReadWriteTestSuite,
    ReadWriteTestSuite,
)

from langchain_community.vectorstores import ApertureDB


class TestApertureDBReadWriteTestSuite(ReadWriteTestSuite):
    @pytest.fixture
    def vectorstore(self) -> ApertureDB:
        descriptor_set = uuid.uuid4().hex  # Fresh descriptor set for each test
        return ApertureDB(
            embeddings=self.get_embeddings(), descriptor_set=descriptor_set
        )


class TestAsyncApertureDBReadWriteTestSuite(AsyncReadWriteTestSuite):
    @pytest.fixture
    async def vectorstore(self) -> ApertureDB:
        descriptor_set = uuid.uuid4().hex  # Fresh descriptor set for each test
        return ApertureDB(
            embeddings=self.get_embeddings(), descriptor_set=descriptor_set
        )
