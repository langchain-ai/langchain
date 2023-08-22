from typing import Any, Callable, Dict, List, Optional

from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.chains.comprehend_moderation import (
    BaseModeration,
    BaseModerationCallbackHandler,
)
from langchain.pydantic_v1 import root_validator


class AmazonComprehendModerationChain(Chain):

    """
    A subclass of Chain, designed to apply moderation to LLMs.

    Attributes:
        - moderation_config (Optional[Dict[str, Any]]): Configuration settings for moderation. Defaults to None.
        - output_key (str): Key used to fetch/store the output in data containers. Defaults to "output".
        - input_key (str): Key used to fetch/store the input in data containers. Defaults to "input".
        - force_base_exception (Optional[bool]): If set to True, enforces the base exception handling. Defaults to False.
        - client (Optional[Any]): Placeholder for a Boto3 client object for connection to Amazon Comprehend.
        - moderation_callback (Optional[BaseModerationCallbackHandler]): Placeholder for a potential callback method or function. Defaults to None.

    Methods:
        __init__: Constructor method for initializing the ModerationChain object.

    Raises:
        - ValueError
        - ModuleNotFoundError
        - BaseModerationError
        - ModerationPiiError
        - ModerationToxicityError
        - ModerationIntentionError

    Note: The `output_key` and `input_key` are considered private and should not be modified directly by external users.
    """

    output_key: str = "output"  #: :meta private:
    input_key: str = "input"  #: :meta private:
    moderation_config: Optional[Dict[str, Any]] = None
    force_base_exception: Optional[bool] = False
    client: Optional[Any]
    moderation_callback: Optional[BaseModerationCallbackHandler] = None
    unique_id: Optional[str] = None

    @root_validator(pre=True)
    def create_client(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Creates an Amazon Comprehend client using Boto3 based on the provided configuration values.

        Args:
            values (Dict[str, Any]): A dictionary containing configuration values.

        Returns:
            Dict[str, Any]: A dictionary with the updated configuration values, including the Amazon Comprehend client.

        Raises:
            ModuleNotFoundError: If the 'boto3' package is not installed.
            ValueError: If there is an issue importing 'boto3' or loading AWS credentials.

        Example usage:
        ===================
        config = {
            "credentials_profile_name": "my-profile",
            "region_name": "us-west-2"
        }
        updated_config = create_client(config)
        comprehend_client = updated_config["client"]
        ==================
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
                "profile name are valid."
            ) from e

    @property
    def output_keys(self) -> List[str]:
        """
        Returns a list of output keys.

        This method defines the output keys that will be used to access the output values produced by
        the chain or function. It ensures that the specified keys are available to access the outputs.

        Returns:
            List[str]: A list of output keys.

        Note:
            This method is considered private and may not be intended for direct external use.

        Example usage:
        ========================
        chain = AmazonComprehendModeration()
        output_keys_list = chain.output_keys()
        print(output_keys_list)
        ========================
        """
        return [self.output_key]

    @property
    def input_keys(self) -> List[str]:
        """
        Returns a list of input keys expected by the prompt.

        This method defines the input keys that the prompt expects in order to perform its processing.
        It ensures that the specified keys are available for providing input to the prompt.

        Returns:
            List[str]: A list of input keys.

        Note:
            This method is considered private and may not be intended for direct external use.

        Example usage:
        ======================
        chain = AmazonComprehendModeration()
        input_keys_list = chain.input_keys()
        print(input_keys_list)
        =======================
        """
        return [self.input_key]

    def _call(
        self,
        inputs: Dict[str, Any] = None,
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        """
        Executes the guardrailed moderation process on the input text and returns the processed output.

        This internal method performs the guardrailed moderation process on the input text. It converts the input prompt
        value to plain text, applies the specified filters, and then converts the filtered output back to a suitable
        prompt value object. Additionally, it provides the option to log information about the run using the provided
        `run_manager`.

        Args:
            inputs (Dict[str, Any], optional): A dictionary containing input values. Default is None.
            run_manager (Optional[CallbackManagerForChainRun], optional): A run manager to handle run-related events. Default is None.

        Returns:
            Dict[str, str]: A dictionary containing the processed output of the guardrailed moderation process.

        Example usage:
        ```
        input_prompt = StringPromptValue(text="Original text.")
        inputs = {"input_key": input_prompt}
        output = self._call(inputs=inputs)
        print(output)  # Outputs the processed output dictionary.
        ```

        Note:
            To log information about the run using the `run_manager`, call methods on it, as demonstrated in the example.

        Raises:
            ValueError: If there is an error during the guardrailed moderation process.
        """
        if run_manager:
            run_manager.on_text(f"Running AmazonComprehendModerationChain...\n")
        moderation = BaseModeration(
            client=self.client,
            config=self.moderation_config,
            force_base_exception=self.force_base_exception,
            moderation_callback=self.moderation_callback,
            unique_id=self.unique_id,
            run_manager=run_manager,
        )
        response = moderation.moderate(prompt=inputs[self.input_keys[0]])
        return {self.output_key: response}
