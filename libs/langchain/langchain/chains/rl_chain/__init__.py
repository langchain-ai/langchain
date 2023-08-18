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
)
from langchain.chains.rl_chain.pick_best_chain import PickBest


def configure_logger():
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
    "Embed",
    "BasedOn",
    "ToSelectFrom",
    "SelectionScorer",
    "AutoSelectionScorer",
    "Embedder",
    "Policy",
    "VwPolicy",
]
