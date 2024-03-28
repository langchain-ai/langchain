from typing import List

from langchain_core.tools import BaseToolkit

from langchain_community.tools import BaseTool
from langchain_community.tools.cogniswitch.tool import (
    CogniswitchKnowledgeRequest,
    CogniswitchKnowledgeSourceFile,
    CogniswitchKnowledgeSourceURL,
    CogniswitchKnowledgeStatus,
)


class CogniswitchToolkit(BaseToolkit):
    """
    Toolkit for CogniSwitch.

    Use the toolkit to get all the tools present in the cogniswitch and
    use them to interact with your knowledge
    """

    cs_token: str  # cogniswitch token
    OAI_token: str  # OpenAI API token
    apiKey: str  # Cogniswitch OAuth token

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        return [
            CogniswitchKnowledgeStatus(
                cs_token=self.cs_token, OAI_token=self.OAI_token, apiKey=self.apiKey
            ),
            CogniswitchKnowledgeRequest(
                cs_token=self.cs_token, OAI_token=self.OAI_token, apiKey=self.apiKey
            ),
            CogniswitchKnowledgeSourceFile(
                cs_token=self.cs_token, OAI_token=self.OAI_token, apiKey=self.apiKey
            ),
            CogniswitchKnowledgeSourceURL(
                cs_token=self.cs_token, OAI_token=self.OAI_token, apiKey=self.apiKey
            ),
        ]
