"""Wrapper arround Google's PaLM Embeddings APIs."""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, root_validator

from langchain.embeddings.base import Embeddings
from langchain.utils import get_from_dict_or_env

from functools import wraps
from google.api_core.exceptions import ResourceExhausted
from google.api_core.exceptions import ServiceUnavailable
import time

# Retry decorator
def retry(ExceptionToCheck, tries=5, delay=4, backoff=2, logger=None):
    """
    Args:
        ExceptionToCheck (Exception or tuple): The exception to check.  may be a tuple of exceptions to check
        tries (int, optional): Number of times to try (not retry) before giving up. Defaults to 4.
        delay (int, optional): Initial delay between retries in seconds Defaults to 3.
        backoff (int, optional): Backoff multiplier e.g. value of 2 will double the delay each retry. Defaults to 2.
        logger (logging.Logger , optional):l Logger to use. If None, print Defaults to None.
    """
    def deco_retry(f):
        @wraps(f)
        def f_retry(*args, **kwargs):
            mtries, mdelay = tries, delay
            while mtries > 1:
                try:
                    return f(*args, **kwargs)
                except ExceptionToCheck:
                    msg = "%s, Retrying in %d seconds..." % (str(ExceptionToCheck), mdelay)
                    print(msg)
                    time.sleep(mdelay)
                    mtries -= 1
                    mdelay *= backoff
            return f(*args, **kwargs)
        return f_retry  # true decorator
    return deco_retry


class GooglePalmEmbeddings(BaseModel, Embeddings):
    client: Any
    google_api_key: Optional[str]
    model_name: str = "models/embedding-gecko-001"

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate api key, python package exists."""
        google_api_key = get_from_dict_or_env(
            values, "google_api_key", "GOOGLE_API_KEY"
        )
        try:
            import google.generativeai as genai

            genai.configure(api_key=google_api_key)
        except ImportError:
            raise ImportError("Could not import google.generativeai python package.")

        values["client"] = genai

        return values

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(text) for text in texts]

    @retry((ResourceExhausted, ServiceUnavailable), tries=5, delay=60, backoff=2)
    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        embedding = self.client.generate_embeddings(self.model_name, text)
        return embedding["embedding"]


