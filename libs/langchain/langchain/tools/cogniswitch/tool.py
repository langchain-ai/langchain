from typing import Any, Dict, Optional

import requests

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools import BaseTool


class CogniswitchAnswerTool(BaseTool):
    """
    A chain class for interacting with the Cogniswitch service to answer questions.
    name: str = "Cogniswitch"
    description: str = (
        "A wrapper around cogniswitch. "
        "Input should be a search query."
    )
    """

    name: str = "CogniswitchAnswerTool"
    description: str = "This tool can be used to get answers using natural language queries from your knowledge sources."
    cs_token: str
    OAI_token: str
    apiKey: str
    api_url = "https://api.cogniswitch.ai:8243/cs-api/0.0.1/cs/knowledgeRequest"

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """
        Execute the chain to answer a query.

        Args:
            query (str): Natural language query,
              that you would like to ask to your knowledge graph.
            run_manager (Optional[CallbackManagerForChainRun]):
            Manager for chain run callbacks.

        Returns:
            Dict[str, Any]: Output dictionary containing
            the 'response' from the service.
        """
        response = self.answer_cs(self.cs_token, self.OAI_token, query, self.apiKey)
        return response

    def answer_cs(self, cs_token: str, OAI_token: str, query: str, apiKey: str) -> dict:
        """
        Send a query to the Cogniswitch service and retrieve the response.

        Args:
            cs_token (str): Cogniswitch token.
            OAI_token (str): OpenAI token.
            apiKey (str): OAuth token.
            query (str): Query to be answered.

        Returns:
            dict: Response JSON from the Cogniswitch service.
        """
        if not cs_token:
            raise ValueError("Missing cs_token")
        if not OAI_token:
            raise ValueError("Missing OpenAI token")
        if not apiKey:
            raise ValueError("Missing cogniswitch OAuth token")
        if not query:
            raise ValueError("Missing input query")

        headers = {
            "apiKey": apiKey,
            "platformToken": cs_token,
            "openAIToken": OAI_token,
        }

        data = {"query": query}
        response = requests.post(self.api_url, headers=headers, verify=False, data=data)
        return response.json()


class CogniswitchStoreTool(BaseTool):
    """
    A chain class for interacting with the Cogniswitch service to store data.
    name: str = "Cogniswitch"
    description: str = (
        "A wrapper around cogniswitch. "
        "Input should be a file path or url."
    )
    """

    name: str = "CogniswitchStoreTool"
    description: str = "This tool can be used to analyze, organize and store your knowledge from documents or urls"
    cs_token: str
    OAI_token: str
    apiKey: str
    knowledgesource_file = (
        "https://api.cogniswitch.ai:8243/cs-api/0.0.1/cs/knowledgeSource/file"
    )
    knowledgesource_url = (
        "https://api.cogniswitch.ai:8243/cs-api/0.0.1/cs/knowledgeSource/url"
    )

    def _run(
        self,
        file: Optional[str] = None,
        url: Optional[str] = None,
        document_name: Optional[str] = None,
        document_description: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """
        Execute the chain to answer a query.

        Args:
            file Optional[str]: The file path of your knowledge
            url Optional[str]: The url of your knowledge
            document_name Optional[str]: Name of your knowledge document
            document_description Optional[str]: Description of your knowledge document
            run_manager (Optional[CallbackManagerForChainRun]):
            Manager for chain run callbacks.

        Returns:
            Dict[str, Any]: Output dictionary containing
            the 'response' from the service.
        """
        if not file:
            file = None
        if not url:
            url = None
        response = self.store_data(
            file=file,
            url=url,
            document_name=document_name,
            document_description=document_description,
        )
        return response

    def store_data(
        self,
        url: Optional[str],
        file: Optional[str],
        document_name: Optional[str],
        document_description: Optional[str],
    ) -> dict:
        """
        Store data using the Cogniswitch service.

        Args:
            url (Optional[str]): URL link.
            file (Optional[str]): file path of your file.
            the current files supported by the files are
            .txt, .pdf, .docx, .doc, .html
            document_name (Optional[str]): Name of the document you are uploading.
            document_description (Optional[str]): Description of the document.

        Returns:
            dict: Response JSON from the Cogniswitch service.
        """
        headers = {
            "apiKey": self.apiKey,
            "openAIToken": self.OAI_token,
            "platformToken": self.cs_token,
        }
        data: Dict[str, Any]
        if not document_name:
            document_name = ""
        if not document_description:
            document_description = ""
        if not file and not url:
            return {
                "message": "No input provided",
            }
        elif file and url:
            return {
                "message": "Too many inputs, please provide either file or url",
            }
        elif url:
            files = None
            data = {"url": url}
            response = requests.post(
                self.knowledgesource_url,
                headers=headers,
                verify=False,
                data=data,
                files=files,
            )

        elif file:
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
                self.knowledgesource_file,
                headers=headers,
                verify=False,
                data=data,
                files=files,
            )
        if response.status_code == 200:
            return response.json()
        else:
            return {"message": "Bad Request"}
