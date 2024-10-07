from typing import List

from langchain_core.tools import BaseTool
from langchain_core.tools.base import BaseToolkit

from langchain_community.tools.cogniswitch.tool import (
    CogniswitchKnowledgeRequest,
    CogniswitchKnowledgeSourceFile,
    CogniswitchKnowledgeSourceURL,
    CogniswitchKnowledgeStatus,
)


class CogniswitchToolkit(BaseToolkit):
    """Toolkit for CogniSwitch.

    Use the toolkit to get all the tools present in the Cogniswitch and
    use them to interact with your knowledge.

    Parameters:
        cs_token: str. The Cogniswitch token.
        OAI_token: str. The OpenAI API token.
        apiKey: str. The Cogniswitch OAuth token.
    """

    cs_token: str
    OAI_token: str
    apiKey: str

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
