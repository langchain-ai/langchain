import json
import logging
from typing import Dict, Optional

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools.slack.base import SlackBaseTool


class SlackGetChannelIdNameDict(SlackBaseTool):
    name: str = "get_channelid_name_dict"
    description: str = "Use this tool to get channelid-name dict."

    def _run(
        self,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        logging.getLogger(__name__)

        result = self.client.conversations_list()
        channelId_Name: Dict[str,str] = {}
        self.save_conversations(result["channels"], channelId_Name)

        json.dumps(channelId_Name)
        return json.dumps(result["channels"])

    @classmethod
    def save_conversations(self, conversations, channelId_Name,
    ) -> None:
        conversation_id = ""
        for conversation in conversations:
            conversation_id = conversation["id"]
            channelId_Name[conversation_id] = conversation["name"]
