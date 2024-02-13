import logging

from langchain_core._api.deprecation import deprecated
from langchain_core.language_models.llms import LLM
logger = logging.getLogger(__name__)

@deprecated("0.1.7", alternative="HuggingFaceEndppoint") 
class HuggingFaceTextGenInference(LLM):
    """
    HuggingFace text generation API.

    Deprecated in favor of HuggingFaceEndpoint to consolidate HuggingFace classes.
    """
