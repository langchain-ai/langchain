import logging
from typing import List

from tests.integration_tests.vectorstores.qdrant.common import qdrant_is_not_running

logger = logging.getLogger(__name__)


def qdrant_locations() -> List[str]:
    if qdrant_is_not_running():
        logger.warning("Running Qdrant async tests in memory mode only.")
        return [":memory:"]
    return ["http://localhost:6333", ":memory:"]
