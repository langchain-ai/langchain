"""Test VDMS functionality."""

from __future__ import annotations

import logging
import os
import uuid

import pytest
from langchain_tests.integration_tests.vectorstores import VectorStoreIntegrationTests

from langchain_community.vectorstores.vdms import VDMS, VDMS_Client

logging.basicConfig(level=logging.DEBUG)


# To spin up a detached VDMS server:
# docker pull intellabs/vdms:latest
# docker run -d -p $VDMS_DBPORT:55555 intellabs/vdms:latest


class TestVDMSStandard(VectorStoreIntegrationTests):
    @pytest.fixture
    def vectorstore(self) -> VDMS:
        test_name = uuid.uuid4().hex
        client = VDMS_Client(
            host=os.getenv("VDMS_DBHOST", "localhost"),
            port=int(os.getenv("VDMS_DBPORT", 6025)),
        )
        return VDMS(
            client=client,
            embedding=self.get_embeddings(),
            collection_name=test_name,
        )
