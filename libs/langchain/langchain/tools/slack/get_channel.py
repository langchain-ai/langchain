import json
import logging
from typing import Optional

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools.slack.base import SlackBaseTool


class SlackGetChannel(SlackBaseTool):
    name: str = "get_channelid_name_dict"
    description: str = "Use this tool to get channelid-name dict."

    def _run(
        self,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        try:
            logging.getLogger(__name__)

            result = self.client.conversations_list()
            channels = result["channels"]
            filtered_result = [
                {key: channel[key] for key in ("id", "name", "created", "num_members")}
                for channel in channels
                if "id" in channel
                and "name" in channel
                and "created" in channel
                and "num_members" in channel
            ]
            return json.dumps(filtered_result)

        except Exception as e:
            return "Error creating conversation: {}".format(e)
