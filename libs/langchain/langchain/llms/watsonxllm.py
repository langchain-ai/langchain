import logging
import os
from typing import Any, Dict, List, Mapping, Optional, Union

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from langchain.pydantic_v1 import Extra, SecretStr, root_validator
from langchain.utils import convert_to_secret_str, get_from_dict_or_env

logger = logging.getLogger(__name__)


class WatsonxLLM(LLM):
    """
    IBM watsonx.ai large language models.

    To use, you should have ``ibm_watson_machine_learning`` python package installed,
    and the environment variable ``WATSONX_APIKEY`` set with your API key, or pass
    it as a named parameter to the constructor.


    Example:
        .. code-block:: python

            from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames
            parameters = {
                GenTextParamsMetaNames.DECODING_METHOD: "sample",
                GenTextParamsMetaNames.MAX_NEW_TOKENS: 100,
                GenTextParamsMetaNames.MIN_NEW_TOKENS: 1,
                GenTextParamsMetaNames.TEMPERATURE: 0.5,
                GenTextParamsMetaNames.TOP_K: 50,
                GenTextParamsMetaNames.TOP_P: 1,
            }

            from langchain.llms import WatsonxLLM
            llm = WatsonxLLM(
                model_id="google/flan-ul2",
                url="https://us-south.ml.cloud.ibm.com",
                apikey="*****",
                project_id="*****",
                params=parameters,
            )
    """

    model_id: str = ""
    """Type of model to use."""

    project_id: str = ""
    """ID of the Watson Studio project."""

    space_id: str = ""
    """ID of the Watson Studio space."""

    url: Optional[SecretStr] = None
    """Url to Watson Machine Learning instance"""

    apikey: Optional[SecretStr] = None
    """Apikey to Watson Machine Learning instance"""

    token: Optional[SecretStr] = None
    """Token to Watson Machine Learning instance"""

    password: Optional[SecretStr] = None
    """Password to Watson Machine Learning instance"""

    username: Optional[SecretStr] = None
    """Username to Watson Machine Learning instance"""

    instance_id: Optional[SecretStr] = None
    """Instance_id of Watson Machine Learning instance"""

    version: Optional[SecretStr] = None
    """Version of Watson Machine Learning instance"""

    params: Optional[dict] = None
    """Model parameters to use during generate requests."""

    verify: Union[str, bool] = ""
    """User can pass as verify one of following:
        the path to a CA_BUNDLE file
        the path of directory with certificates of trusted CAs
        True - default path to truststore will be taken
        False - no verification will be made"""

    watsonx_model: Any

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {
            "url": "WATSONX_URL",
            "apikey": "WATSONX_APIKEY",
            "token": "WATSONX_TOKEN",
            "password": "WATSONX_PASSWORD",
            "username": "WATSONX_USERNAME",
            "instance_id": "WATSONX_INSTANCE_ID",
        }

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that credentials and python package exists in environment."""
        values["url"] = convert_to_secret_str(
            get_from_dict_or_env(values, "url", "WATSONX_URL")
        )
        if "cloud.ibm.com" in values.get("url", "").get_secret_value():
            values["apikey"] = convert_to_secret_str(
                get_from_dict_or_env(values, "apikey", "WATSONX_APIKEY")
            )
        else:
            if (
                not values["token"]
                and "WATSONX_TOKEN" not in os.environ
                and not values["password"]
                and "WATSONX_PASSWORD" not in os.environ
                and not values["apikey"]
                and "WATSONX_APIKEY" not in os.environ
            ):
                raise ValueError(
                    "Did not find 'token', 'password' or 'apikey',"
                    " please add an environment variable"
                    " `WATSONX_TOKEN`, 'WATSONX_PASSWORD' or 'WATSONX_APIKEY' "
                    "which contains it,"
                    " or pass 'token', 'password' or 'apikey'"
                    " as a named parameter."
                )
            elif values["token"] or "WATSONX_TOKEN" in os.environ:
                values["token"] = convert_to_secret_str(
                    get_from_dict_or_env(values, "token", "WATSONX_TOKEN")
                )
            elif values["password"] or "WATSONX_PASSWORD" in os.environ:
                values["password"] = convert_to_secret_str(
                    get_from_dict_or_env(values, "password", "WATSONX_PASSWORD")
                )
                values["username"] = convert_to_secret_str(
                    get_from_dict_or_env(values, "username", "WATSONX_USERNAME")
                )
            elif values["apikey"] or "WATSONX_APIKEY" in os.environ:
                values["apikey"] = convert_to_secret_str(
                    get_from_dict_or_env(values, "apikey", "WATSONX_APIKEY")
                )
                values["username"] = convert_to_secret_str(
                    get_from_dict_or_env(values, "username", "WATSONX_USERNAME")
                )
            if not values["instance_id"] or "WATSONX_INSTANCE_ID" not in os.environ:
                values["instance_id"] = convert_to_secret_str(
                    get_from_dict_or_env(values, "instance_id", "WATSONX_INSTANCE_ID")
                )

        try:
            from ibm_watson_machine_learning.foundation_models import Model

            credentials = {
                "url": values["url"].get_secret_value() if values["url"] else None,
                "apikey": values["apikey"].get_secret_value()
                if values["apikey"]
                else None,
                "token": values["token"].get_secret_value()
                if values["token"]
                else None,
                "password": values["password"].get_secret_value()
                if values["password"]
                else None,
                "username": values["username"].get_secret_value()
                if values["username"]
                else None,
                "instance_id": values["instance_id"].get_secret_value()
                if values["instance_id"]
                else None,
                "version": values["version"].get_secret_value()
                if values["version"]
                else None,
            }
            credentials_without_none_value = {
                key: value for key, value in credentials.items() if value is not None
            }

            watsonx_model = Model(
                model_id=values["model_id"],
                credentials=credentials_without_none_value,
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
            "model_id": self.model_id,
            "params": self.params,
            "project_id": self.project_id,
            "space_id": self.space_id,
        }

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "IBM watsonx.ai"

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
