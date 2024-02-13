from langchain_core.language_models.llms import LLM
from langchain_core._api.deprecation import deprecated


@deprecated("0.1.7", alternative="HuggingFaceEndppoint") 
class HuggingFaceHub(LLM):
    """
    HuggingFace text generation API.

    Deprecated in favor of HuggingFaceEndpoint to consolidate HuggingFace classes.
    """
