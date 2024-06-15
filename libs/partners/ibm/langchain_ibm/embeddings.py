import os
from typing import Dict, List, Optional, Union

from ibm_watsonx_ai import APIClient, Credentials  # type: ignore
from ibm_watsonx_ai.foundation_models.embeddings import Embeddings  # type: ignore
from langchain_core.embeddings import Embeddings as LangChainEmbeddings
from langchain_core.pydantic_v1 import (
    BaseModel,
    Extra,
    Field,
    SecretStr,
    root_validator,
)
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env


class WatsonxEmbeddings(BaseModel, LangChainEmbeddings):
    model_id: str = ""
    """Type of model to use."""

    project_id: str = ""
    """ID of the Watson Studio project."""

    space_id: str = ""
    """ID of the Watson Studio space."""

    url: Optional[SecretStr] = None
    """Url to Watson Machine Learning or CPD instance"""

    apikey: Optional[SecretStr] = None
    """Apikey to Watson Machine Learning or CPD instance"""

    token: Optional[SecretStr] = None
    """Token to CPD instance"""

    password: Optional[SecretStr] = None
    """Password to CPD instance"""

    username: Optional[SecretStr] = None
    """Username to CPD instance"""

    instance_id: Optional[SecretStr] = None
    """Instance_id of CPD instance"""

    version: Optional[SecretStr] = None
    """Version of CPD instance"""

    params: Optional[dict] = None
    """Model parameters to use during generate requests."""

    verify: Union[str, bool, None] = None
    """User can pass as verify one of following:
        the path to a CA_BUNDLE file
        the path of directory with certificates of trusted CAs
        True - default path to truststore will be taken
        False - no verification will be made"""

    watsonx_embed: Embeddings = Field(default=None)  #: :meta private:

    watsonx_client: APIClient = Field(default=None)  #: :meta private:

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that credentials and python package exists in environment."""
        if isinstance(values.get("watsonx_client"), APIClient):
            watsonx_embed = Embeddings(
                model_id=values["model_id"],
                params=values["params"],
                api_client=values["watsonx_client"],
                project_id=values["project_id"],
                space_id=values["space_id"],
                verify=values["verify"],
            )
            values["watsonx_embed"] = watsonx_embed

        else:
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
                        get_from_dict_or_env(
                            values, "instance_id", "WATSONX_INSTANCE_ID"
                        )
                    )

            credentials = Credentials(
                url=values["url"].get_secret_value() if values["url"] else None,
                api_key=values["apikey"].get_secret_value()
                if values["apikey"]
                else None,
                token=values["token"].get_secret_value() if values["token"] else None,
                password=values["password"].get_secret_value()
                if values["password"]
                else None,
                username=values["username"].get_secret_value()
                if values["username"]
                else None,
                instance_id=values["instance_id"].get_secret_value()
                if values["instance_id"]
                else None,
                version=values["version"].get_secret_value()
                if values["version"]
                else None,
                verify=values["verify"],
            )

            watsonx_embed = Embeddings(
                model_id=values["model_id"],
                params=values["params"],
                credentials=credentials,
                project_id=values["project_id"],
                space_id=values["space_id"],
            )

            values["watsonx_embed"] = watsonx_embed

        return values

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        return self.watsonx_embed.embed_documents(texts=texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return self.embed_documents([text])[0]
