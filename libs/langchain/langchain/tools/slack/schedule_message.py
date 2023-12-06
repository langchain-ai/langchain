import logging
from datetime import datetime as dt
from typing import Optional, Type

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools.slack.base import SlackBaseTool
from langchain.tools.slack.utils import UTC_FORMAT

logger = logging.getLogger(__name__)


class ScheduleMessageSchema(BaseModel):
    """Input for ScheduleMessageTool."""

    message: str = Field(
        ...,
        description="The message to be sent.",
    )
    channel: str = Field(
        ...,
        description="The channel, private group, or IM channel to send message to.",
    )
    timestamp: str = Field(
        ...,
        description="The datetime for when the message should be sent in the "
        ' following format: YYYY-MM-DDTHH:MM:SS±hh:mm, where "T" separates the date '
        " and time components, and the time zone offset is specified as ±hh:mm. "
        ' For example: "2023-06-09T10:30:00+03:00" represents June 9th, '
        " 2023, at 10:30 AM in a time zone with a positive offset of 3 "
        " hours from Coordinated Universal Time (UTC).",
    )


class SlackScheduleMessage(SlackBaseTool):
    """Tool for scheduling a message in Slack."""

    name: str = "schedule_message"
    description: str = (
        "Use this tool to schedule a message to be sent on a specific date and time."
    )
    args_schema: Type[ScheduleMessageSchema] = ScheduleMessageSchema

    def _run(
        self,
        message: str,
        channel: str,
        timestamp: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        try:
            unix_timestamp = dt.timestamp(dt.strptime(timestamp, UTC_FORMAT))
            result = self.client.chat_scheduleMessage(
                channel=channel, text=message, post_at=unix_timestamp
            )
            output = "Message scheduled: " + str(result)
            return output
        except Exception as e:
            return "Error scheduling message: {}".format(e)
