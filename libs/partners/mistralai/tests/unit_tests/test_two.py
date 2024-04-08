from ._helper import *
import pytest


@pytest.fixture
def my_fixture():
    return 2


@pytest.fixture
def my_second_fixture():
    return 3
