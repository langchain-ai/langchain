"""**Cross encoders**  are wrappers around cross encoder models
from different APIs and services.

**Cross encoder models** can be LLMs or not.

**Class hierarchy:**

.. code-block::

    CrossEncoder --> <name>CrossEncoder  # Examples: SagemakerEndpointCrossEncoder
"""


import logging

from langchain.cross_encoders.fake import FakeCrossEncoder
from langchain.cross_encoders.huggingface import HuggingFaceCrossEncoder
from langchain.cross_encoders.sagemaker_endpoint import SagemakerEndpointCrossEncoder

logger = logging.getLogger(__name__)

__all__ = [
    "FakeCrossEncoder",
    "HuggingFaceCrossEncoder",
    "SagemakerEndpointCrossEncoder",
]
