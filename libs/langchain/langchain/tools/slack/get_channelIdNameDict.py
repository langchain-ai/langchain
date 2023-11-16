import json
import logging
import os
from typing import List, Optional, Type

from slack_sdk.errors import SlackApiError

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools.slack.base import SlackBaseTool


class SlackGetChannelIdNameDictSchema(BaseModel):
    description: str = (
        "Use this tool to send a message with the provided message fields."
    )


class SlackGetChannelIdNameDict(SlackBaseTool):
    name: str = "get_channelid_name_dict"
    description: str = "Use this tool to get channelid-name dict."
    args_schema: Type[SlackGetChannelIdNameDictSchema] = SlackGetChannelIdNameDictSchema

    def _run(
        self,
        # channelid_name: dict,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> dict:
        logger = logging.getLogger(__name__)

        result = self.client.conversations_list()
        channelId_Name = {}
        self.save_conversations(result["channels"], channelId_Name)

        channelId_Name_json = json.dumps(channelId_Name)
        # print(channelId_Name_json)
        # return channelId_Name_json
        return json.dumps(result["channels"])

        # except SlackApiError as e:
        #     logger.error("Error fetching conversations: {}".format(e))

    # You probably want to use a database to store any conversations information ;)
    # channelId_Name = {}

    def save_conversations(self, conversations, channelId_Name):
        conversation_id = ""
        for conversation in conversations:
            conversation_id = conversation["id"]
            channelId_Name[conversation_id] = conversation["name"]
