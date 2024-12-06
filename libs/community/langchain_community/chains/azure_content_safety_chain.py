"""Pass input through an azure content safety resource."""

from typing import Any, Dict, List, Optional

from langchain.chains.base import Chain
from langchain_core.callbacks import (
    CallbackManagerForChainRun,
)
from langchain_core.exceptions import LangChainException
from langchain_core.utils import get_from_dict_or_env
from pydantic import model_validator


class AzureHarmfulContentError(LangChainException):
    """Exception for handling harmful content detected
    in input for a model or chain according to Azure's
    content safety policy."""

    def __init__(
        self,
        input: str,
    ):
        """Constructor

        Args:
            input (str): The input given by the user to the model.
        """
        self.input = input
        self.message = "The input has breached Azure's Content Safety Policy"
        super().__init__(self.message)


class AzureAIContentSafetyChain(Chain):
    """
    A wrapper for the Azure AI Content Safety API in a Runnable form.
    Allows for harmful content detection and filtering before input is
    provided to a model.

    **Note**:
    This Service will filter input that shows any sign of harmful content,
    this is non-configurable.

    Attributes:
        error (bool): Whether to raise an error if harmful content is detected.
        content_safety_key (Optional[str]): API key for Azure Content Safety.
        content_safety_endpoint (Optional[str]): Endpoint URL for Azure Content Safety.

    Setup:
        1. Follow the instructions here to deploy Azure AI Content Safety:
            https://learn.microsoft.com/azure/ai-services/content-safety/overview

        2. Install ``langchain`` ``langchain_community`` and set the following
        environment variables:

        .. code-block:: bash

            pip install -U langchain langchain-community

            export AZURE_CONTENT_SAFETY_KEY="your-api-key"
            export AZURE_CONTENT_SAFETY_ENDPOINT="https://your-endpoint.azure.com/"


    Example Usage (with safe content):
        .. code-block:: python

            from langchain_community.chains import AzureAIContentSafetyChain
            from langchain_openai import AzureChatOpenAI

            moderate = AzureAIContentSafetyChain()
            prompt = ChatPromptTemplate.from_messages([("system",
                    "repeat after me: {input}")])
            model = AzureChatOpenAI()

            moderated_chain = moderate | prompt | model

            moderated_chain.invoke({"input": "Hey, How are you?"})

    Example Usage (with harmful content):
        .. code-block:: python

            from langchain_community.chains import AzureAIContentSafetyChain
            from langchain_openai import AzureChatOpenAI

            moderate = AzureAIContentSafetyChain()
            prompt = ChatPromptTemplate.from_messages([("system",
                    "repeat after me: {input}")])
            model = AzureChatOpenAI()

            moderated_chain = moderate | prompt | model

            try:
                response = moderated_chain.invoke({"input": "I hate you!"})
            except AzureHarmfulContentError as e:
                print(f'Harmful content: {e.input}')
                raise
    """

    client: Any = None  #: :meta private:
    error: bool = True
    """Whether or not to error if bad content was found."""
    input_key: str = "input"  #: :meta private:
    output_key: str = "output"  #: :meta private:
    content_safety_key: Optional[str] = None
    content_safety_endpoint: Optional[str] = None

    @property
    def input_keys(self) -> List[str]:
        """Expect input key.

        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Return output key.

        :meta private:
        """
        return [self.output_key]

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """Validate that api key and python package exists in environment."""
        content_safety_key = get_from_dict_or_env(
            values, "content_safety_key", "CONTENT_SAFETY_API_KEY"
        )
        content_safety_endpoint = get_from_dict_or_env(
            values, "content_safety_endpoint", "CONTENT_SAFETY_ENDPOINT"
        )
        try:
            import azure.ai.contentsafety as sdk
            from azure.core.credentials import AzureKeyCredential

            values["client"] = sdk.ContentSafetyClient(
                endpoint=content_safety_endpoint,
                credential=AzureKeyCredential(content_safety_key),
            )

        except ImportError:
            raise ImportError(
                "azure-ai-contentsafety is not installed. "
                "Run `pip install azure-ai-contentsafety` to install."
            )
        return values

    def _detect_harmful_content(self, text: str, results: Any) -> str:
        contains_harmful_content = False

        for category in results:
            if category["severity"] > 0:
                contains_harmful_content = True

        if contains_harmful_content:
            error_str = (
                "The input text contains harmful content "
                "according to Azure OpenAI's content policy"
            )
            if self.error:
                raise AzureHarmfulContentError(input=text)
            else:
                return error_str

        return text

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        text = inputs[self.input_key]

        from azure.ai.contentsafety.models import AnalyzeTextOptions

        request = AnalyzeTextOptions(text=text)
        response = self.client.analyze_text(request)

        result = response.categories_analysis
        output = self._detect_harmful_content(text, result)

        return {self.input_key: output, self.output_key: output}
