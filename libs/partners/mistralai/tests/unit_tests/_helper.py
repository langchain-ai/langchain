from typing import Optional


def test_a(my_fixture: int) -> None:
    assert False, f"test_a {my_fixture}"


def test_b(my_fixture: int) -> None:
    assert False, f"test_b {my_fixture}"


def test_c(my_second_fixture: Optional[int]) -> None:
    assert False, f"test_c {my_second_fixture}"
