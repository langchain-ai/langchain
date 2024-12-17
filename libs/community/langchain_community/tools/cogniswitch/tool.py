from __future__ import annotations

from typing import Any, Dict, Optional

import requests
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool


class CogniswitchKnowledgeRequest(BaseTool):  # type: ignore[override]
    """Tool that uses the Cogniswitch service to answer questions.

    name: str = "cogniswitch_knowledge_request"
    description: str = (
        "A wrapper around cogniswitch service to answer the question
        from the knowledge base."
        "Input should be a search query."
    )
    """

    name: str = "cogniswitch_knowledge_request"
    description: str = """A wrapper around cogniswitch service to 
    answer the question from the knowledge base."""
    cs_token: str
    OAI_token: str
    apiKey: str
    api_url: str = "https://api.cogniswitch.ai:8243/cs-api/0.0.1/cs/knowledgeRequest"

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """
        Use the tool to answer a query.

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


class CogniswitchKnowledgeStatus(BaseTool):  # type: ignore[override]
    """Tool that uses the Cogniswitch services to get the
     status of the document or url uploaded.

    name: str = "cogniswitch_knowledge_status"
    description: str = (
        "A wrapper around cogniswitch services to know the status of
         the document uploaded from a url or a file. "
        "Input should be a file name or the url link"
    )
    """

    name: str = "cogniswitch_knowledge_status"
    description: str = """A wrapper around cogniswitch services to know 
    the status of the document uploaded from a url or a file."""
    cs_token: str
    OAI_token: str
    apiKey: str
    knowledge_status_url: str = (
        "https://api.cogniswitch.ai:8243/cs-api/0.0.1/cs/knowledgeSource/status"
    )

    def _run(
        self,
        document_name: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """
        Use the tool to know the status of the document uploaded.

        Args:
            document_name (str): name of the document or
            the url uploaded
            run_manager (Optional[CallbackManagerForChainRun]):
            Manager for chain run callbacks.

        Returns:
            Dict[str, Any]: Output dictionary containing
            the 'response' from the service.
        """
        response = self.knowledge_status(document_name)
        return response

    def knowledge_status(self, document_name: str) -> dict:
        """
        Use this function to know the status of the document or the URL uploaded
        Args:
            document_name (str): The document name or the url that is uploaded.

        Returns:
            dict: Response JSON from the Cogniswitch service.
        """

        params = {"docName": document_name, "platformToken": self.cs_token}
        headers = {
            "apiKey": self.apiKey,
            "openAIToken": self.OAI_token,
            "platformToken": self.cs_token,
        }
        response = requests.get(
            self.knowledge_status_url,
            headers=headers,
            params=params,
            verify=False,
        )
        if response.status_code == 200:
            source_info = response.json()
            source_data = dict(source_info[-1])
            status = source_data.get("status")
            if status == 0:
                source_data["status"] = "SUCCESS"
            elif status == 1:
                source_data["status"] = "PROCESSING"
            elif status == 2:
                source_data["status"] = "UPLOADED"
            elif status == 3:
                source_data["status"] = "FAILURE"
            elif status == 4:
                source_data["status"] = "UPLOAD_FAILURE"
            elif status == 5:
                source_data["status"] = "REJECTED"

            if "filePath" in source_data.keys():
                source_data.pop("filePath")
            if "savedFileName" in source_data.keys():
                source_data.pop("savedFileName")
            if "integrationConfigId" in source_data.keys():
                source_data.pop("integrationConfigId")
            if "metaData" in source_data.keys():
                source_data.pop("metaData")
            if "docEntryId" in source_data.keys():
                source_data.pop("docEntryId")
            return source_data
        else:
            # error_message = response.json()["message"]
            return {
                "message": response.status_code,
            }


class CogniswitchKnowledgeSourceFile(BaseTool):  # type: ignore[override]
    """Tool that uses the Cogniswitch services to store data from file.

    name: str = "cogniswitch_knowledge_source_file"
    description: str = (
        "This calls the CogniSwitch services to analyze & store data from a file.
        If the input looks like a file path, assign that string value to file key.
        Assign document name & description only if provided in input."
    )
    """

    name: str = "cogniswitch_knowledge_source_file"
    description: str = """
        This calls the CogniSwitch services to analyze & store data from a file. 
        If the input looks like a file path, assign that string value to file key. 
        Assign document name & description only if provided in input.
        """
    cs_token: str
    OAI_token: str
    apiKey: str
    knowledgesource_file: str = (
        "https://api.cogniswitch.ai:8243/cs-api/0.0.1/cs/knowledgeSource/file"
    )

    def _run(
        self,
        file: Optional[str] = None,
        document_name: Optional[str] = None,
        document_description: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """
        Execute the tool to store the data given from a file.
        This calls the CogniSwitch services to analyze & store data from a file.
        If the input looks like a file path, assign that string value to file key.
        Assign document name & description only if provided in input.

        Args:
            file Optional[str]: The file path of your knowledge
            document_name Optional[str]: Name of your knowledge document
            document_description Optional[str]: Description of your knowledge document
            run_manager (Optional[CallbackManagerForChainRun]):
            Manager for chain run callbacks.

        Returns:
            Dict[str, Any]: Output dictionary containing
            the 'response' from the service.
        """
        if not file:
            return {
                "message": "No input provided",
            }
        else:
            response = self.store_data(
                file=file,
                document_name=document_name,
                document_description=document_description,
            )
            return response

    def store_data(
        self,
        file: Optional[str],
        document_name: Optional[str],
        document_description: Optional[str],
    ) -> dict:
        """
        Store data using the Cogniswitch service.
        This calls the CogniSwitch services to analyze & store data from a file.
        If the input looks like a file path, assign that string value to file key.
        Assign document name & description only if provided in input.

        Args:
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

        if file is not None:
            files = {"file": open(file, "rb")}

        data = {
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


class CogniswitchKnowledgeSourceURL(BaseTool):  # type: ignore[override]
    """Tool that uses the Cogniswitch services to store data from a URL.

    name: str = "cogniswitch_knowledge_source_url"
    description: str = (
        "This calls the CogniSwitch services to analyze & store data from a url.
        the URL is provided in input, assign that value to the url key.
        Assign document name & description only if provided in input"
    )
    """

    name: str = "cogniswitch_knowledge_source_url"
    description: str = """
    This calls the CogniSwitch services to analyze & store data from a url. 
        the URL is provided in input, assign that value to the url key. 
        Assign document name & description only if provided in input"""
    cs_token: str
    OAI_token: str
    apiKey: str
    knowledgesource_url: str = (
        "https://api.cogniswitch.ai:8243/cs-api/0.0.1/cs/knowledgeSource/url"
    )

    def _run(
        self,
        url: Optional[str] = None,
        document_name: Optional[str] = None,
        document_description: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """
        Execute the tool to store the data given from a url.
        This calls the CogniSwitch services to analyze & store data from a url.
        the URL is provided in input, assign that value to the url key.
        Assign document name & description only if provided in input.

        Args:
            url Optional[str]: The website/url link of your knowledge
            document_name Optional[str]: Name of your knowledge document
            document_description Optional[str]: Description of your knowledge document
            run_manager (Optional[CallbackManagerForChainRun]):
            Manager for chain run callbacks.

        Returns:
            Dict[str, Any]: Output dictionary containing
            the 'response' from the service.
        """
        if not url:
            return {
                "message": "No input provided",
            }
        response = self.store_data(
            url=url,
            document_name=document_name,
            document_description=document_description,
        )
        return response

    def store_data(
        self,
        url: Optional[str],
        document_name: Optional[str],
        document_description: Optional[str],
    ) -> dict:
        """
        Store data using the Cogniswitch service.
        This calls the CogniSwitch services to analyze & store data from a url.
        the URL is provided in input, assign that value to the url key.
        Assign document name & description only if provided in input.

        Args:
            url (Optional[str]): URL link.
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
        if not url:
            return {
                "message": "No input provided",
            }
        else:
            data = {"url": url}
            response = requests.post(
                self.knowledgesource_url,
                headers=headers,
                verify=False,
                data=data,
            )
        if response.status_code == 200:
            return response.json()
        else:
            return {"message": "Bad Request"}
