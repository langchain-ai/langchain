from typing import Any, Dict, List, Optional

from langchain.chains.base import Chain
from langchain_core.callbacks.manager import CallbackManagerForChainRun
from pydantic import model_validator

from langchain_experimental.comprehend_moderation.base_moderation import BaseModeration
from langchain_experimental.comprehend_moderation.base_moderation_callbacks import (
    BaseModerationCallbackHandler,
)
from langchain_experimental.comprehend_moderation.base_moderation_config import (
    BaseModerationConfig,
)


class AmazonComprehendModerationChain(Chain):
    """Moderation Chain, based on `Amazon Comprehend` service.

    See more at https://aws.amazon.com/comprehend/
    """

    output_key: str = "output"  #: :meta private:
    """Key used to fetch/store the output in data containers. Defaults to `output`"""

    input_key: str = "input"  #: :meta private:
    """Key used to fetch/store the input in data containers. Defaults to `input`"""

    moderation_config: BaseModerationConfig = BaseModerationConfig()
    """
    Configuration settings for moderation, 
    defaults to BaseModerationConfig with default values
    """

    client: Optional[Any] = None
    """boto3 client object for connection to Amazon Comprehend"""

    region_name: Optional[str] = None
    """The aws region e.g., `us-west-2`. Fallsback to AWS_DEFAULT_REGION env variable
    or region specified in ~/.aws/config in case it is not provided here.
    """

    credentials_profile_name: Optional[str] = None
    """The name of the profile in the ~/.aws/credentials or ~/.aws/config files, which
    has either access keys or role information specified.
    If not specified, the default credential profile or, if on an EC2 instance,
    credentials from IMDS will be used.
    See: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html
    """

    moderation_callback: Optional[BaseModerationCallbackHandler] = None
    """Callback handler for moderation, this is different 
    from regular callbacks which can be used in addition to this."""

    unique_id: Optional[str] = None
    """A unique id that can be used to identify or group a user or session"""

    @model_validator(mode="before")
    @classmethod
    def create_client(cls, values: Dict[str, Any]) -> Any:
        """
        Creates an Amazon Comprehend client.

        Args:
            values (Dict[str, Any]): A dictionary containing configuration values.

        Returns:
            Dict[str, Any]: A dictionary with the updated configuration values,
                            including the Amazon Comprehend client.

        Raises:
            ModuleNotFoundError: If the 'boto3' package is not installed.
            ValueError: If there is an issue importing 'boto3' or loading
                        AWS credentials.

        Example:
        .. code-block:: python

            config = {
                "credentials_profile_name": "my-profile",
                "region_name": "us-west-2"
            }
            updated_config = create_client(config)
            comprehend_client = updated_config["client"]
        """

        if values.get("client") is not None:
            return values
        try:
            import boto3

            if values.get("credentials_profile_name"):
                session = boto3.Session(profile_name=values["credentials_profile_name"])
            else:
                # use default credentials
                session = boto3.Session()

            client_params = {}
            if values.get("region_name"):
                client_params["region_name"] = values["region_name"]

            values["client"] = session.client("comprehend", **client_params)

            return values
        except ImportError:
            raise ModuleNotFoundError(
                "Could not import boto3 python package. "
                "Please install it with `pip install boto3`."
            )
        except Exception as e:
            raise ValueError(
                "Could not load credentials to authenticate with AWS client. "
                "Please check that credentials in the specified "
                f"profile name are valid. {e}"
            ) from e

    @property
    def output_keys(self) -> List[str]:
        """
        Returns a list of output keys.

        This method defines the output keys that will be used to access the output
        values produced by the chain or function. It ensures that the specified keys
        are available to access the outputs.

        Returns:
            List[str]: A list of output keys.

        Note:
            This method is considered private and may not be intended for direct
            external use.

        """
        return [self.output_key]

    @property
    def input_keys(self) -> List[str]:
        """
        Returns a list of input keys expected by the prompt.

        This method defines the input keys that the prompt expects in order to perform
        its processing. It ensures that the specified keys are available for providing
        input to the prompt.

        Returns:
            List[str]: A list of input keys.

        Note:
            This method is considered private and may not be intended for direct
            external use.
        """
        return [self.input_key]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        """
        Executes the moderation process on the input text and returns the processed
        output.

        This internal method performs the moderation process on the input text. It
        converts the input prompt value to plain text, applies the specified filters,
        and then converts the filtered output back to a suitable prompt value object.
        Additionally, it provides the option to log information about the run using
        the provided `run_manager`.

        Args:
            inputs: A dictionary containing input values
            run_manager: A run manager to handle run-related events. Default is None

        Returns:
            Dict[str, str]: A dictionary containing the processed output of the
                            moderation process.

        Raises:
            ValueError: If there is an error during the moderation process
        """

        if run_manager:
            run_manager.on_text("Running AmazonComprehendModerationChain...\n")

        moderation = BaseModeration(
            client=self.client,
            config=self.moderation_config,
            moderation_callback=self.moderation_callback,
            unique_id=self.unique_id,
            run_manager=run_manager,
        )
        response = moderation.moderate(prompt=inputs[self.input_keys[0]])

        return {self.output_key: response}
