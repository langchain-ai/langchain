import pytest

from ._helper import BaseUnitTests


class TestOne(BaseUnitTests):
    @pytest.fixture
    def my_fixture(self) -> int:
        return 1
