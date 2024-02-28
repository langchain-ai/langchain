import logging

from langchain_experimental.rl_chain.base import (
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
from langchain_experimental.rl_chain.pick_best_chain import (
    PickBest,
    PickBestEvent,
    PickBestFeatureEmbedder,
    PickBestRandomPolicy,
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
    "PickBestFeatureEmbedder",
    "PickBestRandomPolicy",
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
