import logging
import os
from typing import Any, Dict, List, Mapping, Optional, Union

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from langchain.pydantic_v1 import Extra, root_validator
from langchain.utils import get_from_dict_or_env

logger = logging.getLogger(__name__)


class WatsonxLLM(LLM):
    """
    Watsonx large language models.

    To use, you should have ``ibm_watson_machine_learning`` python package installed,
    and the environment variable ``WATSONX_APIKEY`` set with your API key, or pass
    it as a named parameter to the constructor.


    Example:
        .. code-block:: python

            from langchain.llms import WatsonxLLM
            llm = WatsonxLLM(
                model_id='google/flan-ul2',
                credentials = {"url": "https://us-south.ml.cloud.ibm.com",
                                   "apikey": "*****"},
                project_id="*****"
            )
    """

    model_id: str = ""
    """Type of model to use."""

    project_id: str = ""
    """ID of the Watson Studio project."""

    space_id: str = ""
    """ID of the Watson Studio space."""

    credentials: dict = {}
    """Credentials to Watson Machine Learning instance"""

    params: dict = {}
    """Model parameters to use during generate requests."""

    verify: Union[str, bool] = ""
    """User can pass as verify one of following:
        the path to a CA_BUNDLE file
        the path of directory with certificates of trusted CAs
        True - default path to truststore will be taken
        False - no verification will be made"""

    llm_type: str = "IBM watsonx.ai"
    """Type of llm."""

    watsonx_model: Any

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that credentials and python package exists in environment."""
        if "cloud.ibm.com" in values["credentials"].get("url", ""):
            values["credentials"]["apikey"] = get_from_dict_or_env(
                values["credentials"], "apikey", "WATSONX_APIKEY"
            )
        else:
            if (
                "token" not in values["credentials"]
                and "TOKEN" not in os.environ
                and "password" not in values["credentials"]
                and "PASSWORD" not in os.environ
                and "apikey" not in values["credentials"]
                and "APIKEY" not in os.environ
            ):
                raise ValueError(
                    "Did not find 'token', 'password' or 'apikey',"
                    " please add an environment variable"
                    " `TOKEN`, 'PASSWORD' or 'APIKEY' which contains it,"
                    " or pass 'token', 'password' or 'apikey'"
                    " as a named parameter in `credentials`."
                )
            elif "token" in values["credentials"] or "TOKEN" in os.environ:
                values["credentials"]["token"] = get_from_dict_or_env(
                    values["credentials"], "token", "TOKEN"
                )
            elif "password" in values["credentials"] or "PASSWORD" in os.environ:
                values["credentials"]["password"] = get_from_dict_or_env(
                    values["credentials"], "password", "PASSWORD"
                )
                values["credentials"]["username"] = get_from_dict_or_env(
                    values["credentials"], "username", "USERNAME"
                )
            elif "apikey" in values["credentials"] or "APIKEY" in os.environ:
                values["credentials"]["apikey"] = get_from_dict_or_env(
                    values["credentials"], "apikey", "APIKEY"
                )
                values["credentials"]["username"] = get_from_dict_or_env(
                    values["credentials"], "username", "USERNAME"
                )

        try:
            from ibm_watson_machine_learning.foundation_models import Model

            watsonx_model = Model(
                model_id=values["model_id"],
                credentials=values["credentials"],
                params=values["params"],
                project_id=values["project_id"],
                space_id=values["space_id"],
                verify=values["verify"],
            )
            values["watsonx_model"] = watsonx_model

        except ImportError:
            raise ImportError(
                "Could not import ibm_watson_machine_learning python package. "
                "Please install it with `pip install ibm_watson_machine_learning`."
            )
        return values

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            **{"model_id": self.model_id},
            **{"params": self.params},
            **{"project_id": self.project_id},
            **{"space_id": self.space_id},
        }

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return self.llm_type

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the IBM watsonx.ai inference endpoint.

        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            The string generated by the model.

        Example:
            .. code-block:: python

                response = watsonxllm("What is a molecule")
        """

        text = self.watsonx_model.generate_text(prompt=prompt)
        logger.info("Output of watsonx.ai call: {}".format(text))
        if stop is not None:
            text = enforce_stop_tokens(text, stop)
        return text
