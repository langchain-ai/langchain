"""Implements Program-Aided Language Models.

As in https://arxiv.org/pdf/2211.10435.pdf.

This is vulnerable to arbitrary code execution:
https://github.com/hwchase17/langchain/issues/5872
"""
from langchain_experimental.pal_chain.base import PALChain

__all__ = ["PALChain"]
