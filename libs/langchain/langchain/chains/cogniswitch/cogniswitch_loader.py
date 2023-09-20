from typing import Any, Dict, List, Optional

import requests

from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain


class CogniswitchStoreChain(Chain):
    """
    A chain class for storing data using the Cogniswitch service.
    """

    cs_token: str = "cs_token"
    OAI_token: str = "OAI_token"
    url: str = "url"
    file: str = "file"
    apiKey: str = "apiKey"
    document_name: str = "document_name"
    document_description: str = "document_description"
    input_variables = [
        cs_token,
        OAI_token,
        url,
        file,
        document_name,
        document_description,
        apiKey,
    ]
    output_key: str = "response"

    @property
    def input_keys(self) -> List[str]:
        """
        List of expected input keys for the chain.

        Returns:
            List[str]: A list of input keys.
        """
        return [
            self.cs_token,
            self.OAI_token,
            self.file,
            self.url,
            self.document_name,
            self.document_description,
            self.apiKey,
        ]

    @property
    def output_keys(self) -> List[str]:
        """
        List of output keys produced by the chain.

        Returns:
            List[str]: A list of output keys.
        """
        return [self.output_key]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """
        Execute the chain to store data.

        Args:
            inputs (Dict[str, Any]): Input dictionary containing:
            'cs_token', 'OAI_token', 'url', and 'file'.
            run_manager
            (Optional[CallbackManagerForChainRun]): Manager for chain run callbacks.

        Returns:
            Dict[str, Any]: Output dictionary containing the response from the service.
        """
        cs_token = inputs["cs_token"]
        OAI_token = inputs["OAI_token"]
        url = inputs.get("url")
        file = inputs.get("file")
        document_name = inputs.get("document_name")
        document_description = inputs.get("document_description")
        apiKey = inputs["apiKey"]

        if document_name is None:
            document_name = None
        response = self.store_data(
            cs_token, OAI_token, url, file, apiKey, document_name, document_description
        )
        return {"response": response}

    def store_data(
        self,
        cs_token: str,
        OAI_token: str,
        url: Optional[str],
        file: Optional[str],
        apiKey: str,
        document_name: Optional[str],
        document_description: Optional[str],
    ) -> dict:
        """
        Store data using the Cogniswitch service.

        Args:
            cs_token (str): Cogniswitch token.
            OAI_token (str): OpenAI token.
            url (Optional[str]): URL link.
            file (Optional[str]): file path of your file.
            the current files supported by the files are
            .txt, .pdf, .docx, .doc, .html
            document_name (Optional[str]): Name of the document you are uploading.
            document_description (Optional[str]): Description of the document.



        Returns:
            dict: Response JSON from the Cogniswitch service.
        """
        if not document_name:
            document_name = None
        if not document_description:
            document_description = None
        if not file:
            api_url = (
                "https://api.cogniswitch.ai:8243/cs-api/0.0.1/cs/knowledgeSource/url"
            )
            headers = {
                "apiKey": apiKey,
                "openAIToken": OAI_token,
                "platformToken": cs_token,
            }
            files = None
            data = {"url": url}
            response = requests.post(
                api_url, headers=headers, verify=False, data=data, files=files
            )

        if not url:
            api_url = (
                "https://api.cogniswitch.ai:8243/cs-api/0.0.1/cs/knowledgeSource/file"
            )

            headers = {
                "apiKey": apiKey,
                "openAIToken": OAI_token,
                "platformToken": cs_token,
            }
            if file is not None:
                files = {"file": open(file, "rb")}
            else:
                files = None
            data = {
                "url": url,
                "documentName": document_name,
                "documentDescription": document_description,
            }
            response = requests.post(
                api_url, headers=headers, verify=False, data=data, files=files
            )
        if response.status_code == 200:
            return response.json()
        else:
            # error_message = response.json()["message"]
            return {
                "message": "Bad Request",
            }

    def _validate_inputs(self, inputs: Dict[str, Any]) -> None:
        """
        Validate if all required input keys are provided.

        Args:
            inputs (Dict[str, Any]): Input dictionary containing the provided keys.

        Raises:
            ValueError: If any required input key is missing.
        """

        required_keys = {
            self.cs_token,
            self.OAI_token,
            self.file
            if not inputs.get("url")
            else None,  # Either file or url is required
            self.url if not inputs.get("file") else None,
            self.apiKey,
        }

        missing_keys = required_keys.difference(inputs)

        if None not in missing_keys:
            raise ValueError(f"Missing: {missing_keys}")
