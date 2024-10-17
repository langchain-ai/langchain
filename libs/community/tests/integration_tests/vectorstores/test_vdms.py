"""Test VDMS functionality."""

from __future__ import annotations

import logging
import os
import uuid

import pytest
from langchain_standard_tests.integration_tests.vectorstores import (
    AsyncReadWriteTestSuite,
    ReadWriteTestSuite,
)

from langchain_community.vectorstores.vdms import VDMS, VDMS_Client

logging.basicConfig(level=logging.DEBUG)


# The connection string matches the default settings in the docker-compose file
# located in the root of the repository: [root]/docker/docker-compose.yml
# To spin up a detached VDMS server:
# cd [root]/docker
# docker compose up -d vdms


class TestVDMSReadWriteTestSuite(ReadWriteTestSuite):
    @pytest.fixture
    def vectorstore(self) -> VDMS:
        test_name = uuid.uuid4().hex
        client = VDMS_Client(
            host=os.getenv("VDMS_DBHOST", "localhost"),
            port=int(os.getenv("VDMS_DBPORT", 6025)),
        )
        return VDMS(
            client=client, embedding=self.get_embeddings(), collection_name=test_name
        )


class TestAsyncVDMSReadWriteTestSuite(AsyncReadWriteTestSuite):
    @pytest.fixture
    async def vectorstore(self) -> VDMS:
        test_name = uuid.uuid4().hex
        client = VDMS_Client(
            host=os.getenv("VDMS_DBHOST", "localhost"),
            port=int(os.getenv("VDMS_DBPORT", 6025)),
        )
        return VDMS(
            client=client, embedding=self.get_embeddings(), collection_name=test_name
        )
