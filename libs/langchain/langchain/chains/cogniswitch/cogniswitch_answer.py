from typing import Any, Dict, List, Optional

import requests

from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain


class CogniswitchAnswerChain(Chain):
    """
    A chain class for interacting with the Cogniswitch service to answer questions.
    """

    @property
    def input_keys(self) -> List[str]:
        """
        List of expected input keys for the chain.

        Returns:
            List[str]: A list of input keys.
        """
        return ["cs_token", "OAI_token", "query", "apiKey"]

    @property
    def output_keys(self) -> List[str]:
        """
        List of output keys produced by the chain.

        Returns:
            List[str]: A list of output keys.
        """
        return ["response"]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """
        Execute the chain to answer a query.

        Args:
            inputs (Dict[str, Any]): Input dictionary containing
            'cs_token', 'OAI_token', and 'query'.
            run_manager (Optional[CallbackManagerForChainRun]):
            Manager for chain run callbacks.

        Returns:
            Dict[str, Any]: Output dictionary containing
            the 'response' from the service.
        """
        cs_token = inputs["cs_token"]
        OAI_token = inputs["OAI_token"]
        query = inputs["query"]
        apiKey = inputs["apiKey"]
        response = self.answer_cs(cs_token, OAI_token, query, apiKey)
        return {"response": response}

    def answer_cs(self, cs_token: str, OAI_token: str, query: str, apiKey: str) -> dict:
        """
        Send a query to the Cogniswitch service and retrieve the response.

        Args:
            cs_token (str): Cogniswitch token.
            OAI_token (str): OpenAI token.
            query (str): Query to be answered.

        Returns:
            dict: Response JSON from the Cogniswitch service.
        """
        api_url = "https://api.cogniswitch.ai:8243/cs-api/0.0.1/cs/knowledgeRequest"

        headers = {
            "apiKey": apiKey,
            "platformToken": cs_token,
            "openAIToken": OAI_token,
        }

        data = {"query": query}
        response = requests.post(api_url, headers=headers, verify=False, data=data)
        return response.json()

    def _validate_inputs(self, inputs: Dict[str, Any]) -> None:
        """
        Validate if all required input keys are provided.

        Args:
            inputs (Dict[str, Any]): Input dictionary containing the provided keys.

        Raises:
            ValueError: If any required input key is missing.
        """
        missing_keys = set(self.input_keys).difference(inputs)

        if missing_keys:
            raise ValueError(f"Missing {missing_keys}")
