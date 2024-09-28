"""Pass input through an azure content safety resource."""

import os
from typing import Any, Dict, List, Optional

from langchain.chains.base import Chain
from langchain_core.callbacks import (
    CallbackManagerForChainRun,
)
from langchain_core.messages import AIMessage


class AzureOpenAIContentSafetyChain(Chain):
    client: Any  #: :meta private:
    error: bool = False
    """Whether or not to error if bad content was found."""
    input_key: str = "input"  #: :meta private:
    output_key: str = "output"  #: :meta private:

    def __init__(
        self,
        *,
        content_safety_key: Optional[str] = None,
        content_safety_endpoint: Optional[str] = None,
    ) -> None:
        self.content_safety_key = content_safety_key or os.environ["CONTENT_SAFETY_API_KEY"]
        self.content_safety_endpoint = (
            content_safety_endpoint or os.environ["CONTENT_SAFETY_ENDPOINT"]
        )
        try:
            import azure.ai.contentsafety as sdk
            from azure.core.credentials import AzureKeyCredential

            self.client = sdk.ContentSafetyClient(
                endpoint=content_safety_endpoint,
                credential=AzureKeyCredential(content_safety_key),
            )

        except ImportError:
            raise ImportError(
                "azure-ai-contentsafety is not installed. "
                "Run `pip install azure-ai-contentsafety` to install."
            )

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

    def _sentiment_analysis(self, text: str, results: Any) -> str:
        contains_harmful_content = False

        for category in results:
            if category["severity"] > 0:
                contains_harmful_content = True

        if contains_harmful_content:
            error_str = """The input text contains harmful content 
            according to Azure OpenAI's content policy"""
            if self.error:
                raise ValueError(error_str)
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

        request = (
            AnalyzeTextOptions(text=text.content)
            if isinstance(text, AIMessage)
            else AnalyzeTextOptions(text=text)
        )

        response = self.client.analyze_text(request)
        result = response.categories_analysis
        output = self._sentiment_analysis(text, result)

        return {self.output_key: output}
