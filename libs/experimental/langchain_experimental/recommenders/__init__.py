"""**Amazon Personalize** primitives.

[Amazon Personalize](https://docs.aws.amazon.com/personalize/latest/dg/what-is-personalize.html) 
is a fully managed machine learning service that uses your data to generate 
item recommendations for your users.
"""
from langchain_experimental.recommenders.amazon_personalize import AmazonPersonalize
from langchain_experimental.recommenders.amazon_personalize_chain import (
    AmazonPersonalizeChain,
)

__all__ = ["AmazonPersonalize", "AmazonPersonalizeChain"]
