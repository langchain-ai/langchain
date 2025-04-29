import logging
import os

from langchain_qdrant.qdrant import RetrievalMode
from tests.integration_tests.common import qdrant_running_locally

logger = logging.getLogger(__name__)


def qdrant_locations(use_in_memory: bool = True) -> list[str]:
    locations = []

    if use_in_memory:
        logger.info("Running Qdrant tests with in-memory mode.")
        locations.append(":memory:")

    if qdrant_running_locally():
        logger.info("Running Qdrant tests with local Qdrant instance.")
        locations.append("http://localhost:6333")

    if qdrant_url := os.getenv("QDRANT_URL"):
        logger.info(f"Running Qdrant tests with Qdrant instance at {qdrant_url}.")
        locations.append(qdrant_url)

    return locations


def retrieval_modes(
    *, dense: bool = True, sparse: bool = True, hybrid: bool = True
) -> list[RetrievalMode]:
    modes = []

    if dense:
        modes.append(RetrievalMode.DENSE)

    if sparse:
        modes.append(RetrievalMode.SPARSE)

    if hybrid:
        modes.append(RetrievalMode.HYBRID)

    return modes
