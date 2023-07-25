"""Wrapper around EdeneAI API."""
import logging
from typing import Any, Dict, List, Optional

from pydantic import Extra, Field, root_validator

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.utils import get_from_dict_or_env
from langchain.requests import Requests
import json
logger = logging.getLogger(__name__)

class EdenAI(LLM):
    """Wrapper around edenai models.

    To use, you should have
    the environment variable ``EDENAI_API_KEY`` set with your API token.
    You can find your token here: https://app.edenai.run/admin/account/settings

    feature and sub_feature are required, but any other model parameters can also be passed in with the format params={model_param: value, ...}

    for exemple check edenai documentation.
    """
    feature: str
    """ what feature to use """
    sub_feature: str
    """ what subfeature to use """
    
    base_url = "https://api.edenai.run/v2"
    """ base url for edenai api"""
    
    params: Dict[str, Any] = Field(default_factory=dict)
    """" parameters when calling the model """
    
    
    edenai_api_key: Optional[str] = None
    """ api key """
    
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """extra parameters"""
    
    keys=['generated_text','result','items']
    """json formatting keys"""
    

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        edenai_api_key = get_from_dict_or_env(
            values, "edenai_api_key", "EDENAI_API_KEY"
        )
        values["edenai_api_key"] = edenai_api_key
        return values
    
    
    @root_validator(pre=True)
    def build_extra(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = {field.alias for field in cls.__fields__.values()}

        extra = values.get("model_kwargs", {})
        for field_name in list(values):
            if field_name not in all_required_field_names:
                if field_name in extra:
                    raise ValueError(f"Found {field_name} supplied twice.")
                logger.warning(
                    f"""{field_name} was transferred to model_kwargs.
                    Please confirm that {field_name} is what you intended."""
                )
                extra[field_name] = values.pop(field_name)
        values["model_kwargs"] = extra
        return values
    

    @property
    def _llm_type(self) -> str:
        """Return type of model."""
        return "edenai"
    

    def _format_output(self,result) -> str:
        """find the correct json format"""
        for key in self.keys:
            try :
                return(result[self.params["providers"]][key])
            except:
                pass
            
        raise ValueError("key does not exist")
            
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        
        """Call out to EDENAI's complete endpoint.

        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            str response .

        """

        headers = {"Authorization": f"Bearer {self.edenai_api_key}"}
        url = f"{self.base_url}/{self.feature}/{self.sub_feature}"

        payload={**self.params,"text":prompt,**kwargs}
        request=Requests(headers=headers)
        response = request.post(
            url=url,
            data=payload,
        )
        
        if response.status_code != 200:
            raise ValueError(
                f"EDENAI complete call failed with status code {response.status_code}."
            )
            
        result = json.loads(response.text)
        output=self._format_output(result)
        
        if type(output) != str:
            output=json.dumps(output)
            
            
        return output