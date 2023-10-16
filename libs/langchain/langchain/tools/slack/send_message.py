
import logging
import os
# Import WebClient from Python SDK (github.com/slackapi/python-slack-sdk)


from typing import List, Optional, Type

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools.slack.base import SlackBaseTool


class SendMessageSchema(BaseModel):
    """Input for SendMessageTool."""

    text: str = Field(
        ...,
        description="The message text to be sent.",
    )
    channel_id: str = Field(
        ...,
        description="The channel the text to be sent to.",
    )
    

class SlackSendMessage(SlackBaseTool):
    """Tool for sending an email in Office 365."""

    name: str = "send_message"
    description: str = (
        "Use this tool to send a message with the provided message text to the provided channel."
    )
    args_schema: Type[SendMessageSchema] = SendMessageSchema

    def _run(
        self,
        text: str,
        channel_id:str,
        # to: List[str],
        # subject: str,
        # cc: Optional[List[str]] = None,
        # bcc: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        # # Get mailbox object
        # mailbox = self.account.mailbox()
        # message = mailbox.new_message()

        # # Assign message values
        # message.body = body
        # message.subject = subject
        # message.to.add(to)
        # if cc is not None:
        #     message.cc.add(cc)
        # if bcc is not None:
        #     message.bcc.add(cc)

        # message.send()
        # ID of the channel you want to send the message to


        result = self.client.chat_postMessage(
                channel=channel_id, 
                text="test post msg user with .py"
            )
            # logger.info(result)
        output = "Message sent: " + str(text)
        return output
    

        # try:
        # # Call the chat.postMessage method using the WebClient
        #     result = self.client.chat_postMessage(
        #         channel=channel_id, 
        #         text="test post msg user with .py"
        #     )
        #     # logger.info(result)
        #     output = "Message sent: " + str(text)
        #     return output


        # except SlackApiError as e:
        #     return "Message sent failed"
        #     # logger.error(f"Error posting message: {e}")


        


























# from typing import List, Optional, Type

# from langchain.callbacks.manager import CallbackManagerForToolRun
# from langchain.pydantic_v1 import BaseModel, Field
# from langchain.tools.office365.base import O365BaseTool


# class SendMessageSchema(BaseModel):
#     """Input for SendMessageTool."""

#     body: str = Field(
#         ...,
#         description="The message body to be sent.",
#     )
#     to: List[str] = Field(
#         ...,
#         description="The list of recipients.",
#     )
#     subject: str = Field(
#         ...,
#         description="The subject of the message.",
#     )
#     cc: Optional[List[str]] = Field(
#         None,
#         description="The list of CC recipients.",
#     )
#     bcc: Optional[List[str]] = Field(
#         None,
#         description="The list of BCC recipients.",
#     )


# class O365SendMessage(O365BaseTool):
#     """Tool for sending an email in Office 365."""

#     name: str = "send_email"
#     description: str = (
#         "Use this tool to send an email with the provided message fields."
#     )
#     args_schema: Type[SendMessageSchema] = SendMessageSchema

#     def _run(
#         self,
#         body: str,
#         to: List[str],
#         subject: str,
#         cc: Optional[List[str]] = None,
#         bcc: Optional[List[str]] = None,
#         run_manager: Optional[CallbackManagerForToolRun] = None,
#     ) -> str:
#         # Get mailbox object
#         mailbox = self.account.mailbox()
#         message = mailbox.new_message()

#         # Assign message values
#         message.body = body
#         message.subject = subject
#         message.to.add(to)
#         if cc is not None:
#             message.cc.add(cc)
#         if bcc is not None:
#             message.bcc.add(cc)

#         message.send()

#         output = "Message sent: " + str(message)
#         return output
