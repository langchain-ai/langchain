"""Integration test for SerpAPI."""
from langchain.chains.serpapi import SerpAPIChain

def test_call():
    chain = SerpAPIChain()
    output = chain.search("What was Obama's first name?")
    breakpoint()