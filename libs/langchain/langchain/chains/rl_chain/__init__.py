import logging

from langchain.chains.rl_chain.base import (
    AutoSelectionScorer,
    BasedOn,
    Embed,
    Embedder,
    Policy,
    SelectionScorer,
    ToSelectFrom,
    VwPolicy,
    embed,
    stringify_embedding,
)
from langchain.chains.rl_chain.pick_best_chain import (
    PickBest,
    PickBestEvent,
    PickBestSelected,
)


def configure_logger() -> None:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)


configure_logger()

__all__ = [
    "PickBest",
    "PickBestEvent",
    "PickBestSelected",
    "Embed",
    "BasedOn",
    "ToSelectFrom",
    "SelectionScorer",
    "AutoSelectionScorer",
    "Embedder",
    "Policy",
    "VwPolicy",
    "embed",
    "stringify_embedding",
]
