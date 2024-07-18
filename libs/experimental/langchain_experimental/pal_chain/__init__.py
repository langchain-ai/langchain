"""**PAL Chain** implements **Program-Aided Language** Models.

See the paper: https://arxiv.org/pdf/2211.10435.pdf.

This chain is vulnerable to [arbitrary code execution](https://github.com/langchain-ai/langchain/issues/5872).
"""

from langchain_experimental.pal_chain.base import PALChain

__all__ = ["PALChain"]
