from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Mapping, Optional

from langchain_core.embeddings import Embeddings
from langchain_core.utils import pre_init
from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    import oci

CUSTOM_ENDPOINT_PREFIX = "ocid1.generativeaiendpoint"


class OCIAuthType(Enum):
    """OCI authentication types as enumerator."""

    API_KEY = 1
    SECURITY_TOKEN = 2
    INSTANCE_PRINCIPAL = 3
    RESOURCE_PRINCIPAL = 4


class OCIGenAIEmbeddings(BaseModel, Embeddings):
    """OCI embedding models.

    To authenticate, the OCI client uses the methods described in
    https://docs.oracle.com/en-us/iaas/Content/API/Concepts/sdk_authentication_methods.htm

    The authentifcation method is passed through auth_type and should be one of:
    API_KEY (default), SECURITY_TOKEN, INSTANCE_PRINCIPLE, RESOURCE_PRINCIPLE

    Make sure you have the required policies (profile/roles) to
    access the OCI Generative AI service. If a specific config profile is used,
    you must pass the name of the profile (~/.oci/config) through auth_profile.
    If a specific config file location is used, you must pass
    the file location where profile name configs present
    through auth_file_location

    To use, you must provide the compartment id
    along with the endpoint url, and model id
    as named parameters to the constructor.

    Example:
        .. code-block:: python

            from langchain.embeddings import OCIGenAIEmbeddings

            embeddings = OCIGenAIEmbeddings(
                model_id="MY_EMBEDDING_MODEL",
                service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
                compartment_id="MY_OCID"
            )
    """

    client: Any = None  #: :meta private:

    service_models: Any = None  #: :meta private:

    auth_type: Optional[str] = "API_KEY"
    """Authentication type, could be 

    API_KEY, 
    SECURITY_TOKEN, 
    INSTANCE_PRINCIPLE, 
    RESOURCE_PRINCIPLE
    
    If not specified, API_KEY will be used
    """

    auth_profile: Optional[str] = "DEFAULT"
    """The name of the profile in ~/.oci/config
    If not specified , DEFAULT will be used 
    """

    auth_file_location: Optional[str] = "~/.oci/config"
    """Path to the config file.
    If not specified, ~/.oci/config will be used
    """

    model_id: Optional[str] = None
    """Id of the model to call, e.g., cohere.embed-english-light-v2.0"""

    model_kwargs: Optional[Dict] = None
    """Keyword arguments to pass to the model"""

    service_endpoint: Optional[str] = None
    """service endpoint url"""

    compartment_id: Optional[str] = None
    """OCID of compartment"""

    truncate: Optional[str] = "END"
    """Truncate embeddings that are too long from start or end ("NONE"|"START"|"END")"""

    batch_size: int = 96
    """Batch size of OCI GenAI embedding requests. OCI GenAI may handle up to 96 texts
     per request"""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    @pre_init
    def validate_environment(cls, values: Dict) -> Dict:  # pylint: disable=no-self-argument
        """Validate that OCI config and python package exists in environment."""

        # Skip creating new client if passed in constructor
        if values["client"] is not None:
            return values

        try:
            import oci

            client_kwargs = {
                "config": {},
                "signer": None,
                "service_endpoint": values["service_endpoint"],
                "retry_strategy": oci.retry.DEFAULT_RETRY_STRATEGY,
                "timeout": (10, 240),  # default timeout config for OCI Gen AI service
            }

            if values["auth_type"] == OCIAuthType(1).name:
                client_kwargs["config"] = oci.config.from_file(
                    file_location=values["auth_file_location"],
                    profile_name=values["auth_profile"],
                )
                client_kwargs.pop("signer", None)
            elif values["auth_type"] == OCIAuthType(2).name:

                def make_security_token_signer(
                    oci_config: dict[str, Any],
                ) -> "oci.auth.signers.SecurityTokenSigner":
                    pk = oci.signer.load_private_key_from_file(
                        oci_config.get("key_file"), None
                    )
                    with open(
                        str(oci_config.get("security_token_file")), encoding="utf-8"
                    ) as f:
                        st_string = f.read()
                    return oci.auth.signers.SecurityTokenSigner(st_string, pk)

                client_kwargs["config"] = oci.config.from_file(
                    file_location=values["auth_file_location"],
                    profile_name=values["auth_profile"],
                )
                client_kwargs["signer"] = make_security_token_signer(
                    oci_config=client_kwargs["config"]
                )
            elif values["auth_type"] == OCIAuthType(3).name:
                client_kwargs["signer"] = (
                    oci.auth.signers.InstancePrincipalsSecurityTokenSigner()
                )
            elif values["auth_type"] == OCIAuthType(4).name:
                client_kwargs["signer"] = (
                    oci.auth.signers.get_resource_principals_signer()
                )
            else:
                raise ValueError("Please provide valid value to auth_type")

            values["client"] = oci.generative_ai_inference.GenerativeAiInferenceClient(
                **client_kwargs
            )

        except ImportError as ex:
            raise ImportError(
                "Could not import oci python package. "
                "Please make sure you have the oci package installed."
            ) from ex
        except Exception as e:
            raise ValueError(
                """Could not authenticate with OCI client.
                If INSTANCE_PRINCIPAL or RESOURCE_PRINCIPAL is used,
                please check the specified
                auth_profile, auth_file_location and auth_type are valid.""",
                e,
            ) from e

        return values

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        _model_kwargs = self.model_kwargs or {}
        return {
            **{"model_kwargs": _model_kwargs},
        }

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Call out to OCIGenAI's embedding endpoint.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        from oci.generative_ai_inference import models

        if not self.model_id:
            raise ValueError("Model ID is required to embed documents")

        if self.model_id.startswith(CUSTOM_ENDPOINT_PREFIX):
            serving_mode = models.DedicatedServingMode(endpoint_id=self.model_id)
        else:
            serving_mode = models.OnDemandServingMode(model_id=self.model_id)

        embeddings = []

        def split_texts() -> Iterator[List[str]]:
            for i in range(0, len(texts), self.batch_size):
                yield texts[i : i + self.batch_size]

        for chunk in split_texts():
            invocation_obj = models.EmbedTextDetails(
                serving_mode=serving_mode,
                compartment_id=self.compartment_id,
                truncate=self.truncate,
                inputs=chunk,
            )
            response = self.client.embed_text(invocation_obj)
            embeddings.extend(response.data.embeddings)

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Call out to OCIGenAI's embedding endpoint.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        return self.embed_documents([text])[0]
