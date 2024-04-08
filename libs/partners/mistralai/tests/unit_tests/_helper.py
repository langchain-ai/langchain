from abc import ABC, abstractmethod
from typing import Optional

import pytest


class BaseUnitTests(ABC):
    @abstractmethod
    @pytest.fixture
    def my_fixture(self) -> int:
        ...

    @pytest.fixture
    def my_second_fixture(self) -> Optional[int]:
        return None  # default value

    def test_a(self, my_fixture: int) -> None:
        assert False, f"test_a {my_fixture}"

    def test_b(self, my_fixture: int) -> None:
        assert False, f"test_b {my_fixture}"

    def test_c(self, my_second_fixture: Optional[int]) -> None:
        assert False, f"test_c {my_second_fixture}"
