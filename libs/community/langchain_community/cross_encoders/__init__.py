"""**Cross encoders**  are wrappers around cross encoder models from different APIs and
    services.

**Cross encoder models** can be LLMs or not.

**Class hierarchy:**

.. code-block::

    BaseCrossEncoder --> <name>CrossEncoder  # Examples: SagemakerEndpointCrossEncoder
"""


import logging

from langchain_community.cross_encoders.base import BaseCrossEncoder
from langchain_community.cross_encoders.fake import FakeCrossEncoder
from langchain_community.cross_encoders.huggingface import HuggingFaceCrossEncoder
from langchain_community.cross_encoders.sagemaker_endpoint import (
    SagemakerEndpointCrossEncoder,
)

logger = logging.getLogger(__name__)

__all__ = [
    "BaseCrossEncoder",
    "FakeCrossEncoder",
    "HuggingFaceCrossEncoder",
    "SagemakerEndpointCrossEncoder",
]
