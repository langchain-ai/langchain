import os

import pytest

ABS_PATH = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope="module")
def vcr_cassette_dir(request):
    # Put all the cassettes in a directory named 'vhs' in the same directory as the test file
    return os.path.join(
        "vhs", ABS_PATH, request.module.__name__, request.function.__name__ + ".yaml"
    )


@pytest.fixture(scope="module")
def vcr_config():
    return {
        "filter_headers": [("authorization", "DUMMY")],
    }
