"""Salesforce XGen integration."""
import logging
from typing import Any, Dict, Generator, List, Optional

from pydantic import Field, root_validator

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM

logger = logging.getLogger(__name__)


class SalesforceXGen(LLM):
    """Integrating Salesforce XGen model.
    
    Example:
        from langchain.llms import SalesforceXGen
        llm = SalesforceXGen(model_path="/path/to/XGen/model")
    """

    client: Any  #: :meta private:
    model_path: str
    """The path to the XGen model file."""

    xgen_base: Optional[str] = None
    """The path to the Salesforce XGen base model."""

    stop: Optional[List[str]] = []
    """A list of strings to stop generation when encountered."""

    suffix: Optional[str] = Field(None)
    """A suffix to append to the generated text. If None, no suffix is appended."""

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate model path."""
        model_path = values["model_path"]
        
        try:
            from transformers import AutoTokenizer

            values["client"] = AutoTokenizer.from_pretrained(model_path)
        except ImportError:
            raise ModuleNotFoundError(
                "Could not import transformers library. "
                "Please install the transformers library to "
                "use this embedding model: pip install transformers"
            )
        except Exception as e:
            raise ValueError(
                f"Could not load XGen model from path: {model_path}. "
                f"Received error {e}"
            )

        return values

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling module."""
        return {
            "suffix": self.suffix,
            "stop_sequences": self.stop,  # key here is convention among LLM classes
        }

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {**{"model_path": self.model_path}, **self._default_params}

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "salesforce_xgen"

    def _get_parameters(self) -> Dict[str, Any]:
        
        params = self._default_params

        return params

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the XGen model and return the output.

        Args:
            prompt: The prompt to use for generation.

        Returns:
            The generated text.
        """
        params = self._get_parameters()
        params = {**params, **kwargs}
        result = self.client(prompt=prompt, **params)
        return result # adjust for XGen
