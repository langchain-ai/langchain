from langchain.chains.rl_chain.pick_best_chain import PickBest
from langchain.chains.rl_chain.rl_chain_base import (
    Embed,
    BasedOn,
    ToSelectFrom,
    SelectionScorer,
    AutoSelectionScorer,
    Embedder,
    Policy,
    VwPolicy,
)

import logging


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
