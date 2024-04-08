import pytest

from ._helper import BaseUnitTests


class TestTwo(BaseUnitTests):
    @pytest.fixture
    def my_fixture(self) -> int:
        return 2

    @pytest.fixture
    def my_second_fixture(self) -> int:
        return 3
