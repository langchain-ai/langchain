import logging

import pytest

from langchain.vectorstores import Chroma
from tests.integration_tests.vectorstores_new.basic import (
    BaseVectorStoreStaticTestLocal,
    BaseVectorStoreStaticTestRemote,
)

logger = logging.getLogger(__name__)
vector_store_class = Chroma

"""
pytest --capture=no --log-cli-level=DEBUG -vvv tests/integration_tests/vectorstores_new/test_chroma.py 
"""  # noqa: E501


class TestChromaVectorStoreLocalStaticPersistent(BaseVectorStoreStaticTestLocal):
    vector_store_class = vector_store_class

    @classmethod
    def setup_class(cls) -> None:
        pass

    @classmethod
    def teardown_class(cls) -> None:
        pass

    def setup_method(self) -> None:
        pass

    def teardown_method(self) -> None:
        pass


class TestChromaVectorStoreLocalStaticMemory(BaseVectorStoreStaticTestLocal):
    vector_store_class = vector_store_class

    @classmethod
    def setup_class(cls) -> None:
        pass

    @classmethod
    def teardown_class(cls) -> None:
        pass

    def setup_method(self) -> None:
        pass

    def teardown_method(self) -> None:
        pass


@pytest.mark.skip("need docker-compose file to tests it")
class TestChromaStaticTestRemote(BaseVectorStoreStaticTestRemote):
    vector_store_class = vector_store_class

    @classmethod
    def setup_class(cls) -> None:
        pass

    @classmethod
    def teardown_class(cls) -> None:
        pass

    def setup_method(self) -> None:
        pass

    def teardown_method(self) -> None:
        pass
