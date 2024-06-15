from typing import Dict

import pytest


@pytest.fixture(scope="module")
def vcr_config() -> Dict:
    return {
        # IMPORTANT: Filter out the Authorization header from stored replay test data.
        "filter_headers": [("Authorization", None)],
        "ignore_hosts": ["storage.googleapis.com"],
    }
