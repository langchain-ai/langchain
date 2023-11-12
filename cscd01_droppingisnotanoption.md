# Issue

For our [issue](https://github.com/langchain-ai/langchain/issues/11747), we will be working on implementing a Slack Toolkit for LangChain. 

# Mock Solution

`toolkit.py` in (`/workspaces/langchain/libs/langchain/langchain/agents/agent_toolkits/slack/toolkit.py`)
```
class SlackToolkit(BaseToolkit):
    """Toolkit for interacting with Slack."""

    client: WebClient=Field(default_factory=authenticate)
    
    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        return [SlackSendMessage(), 
                SlackGetMessage()]
```

`base.py` (in `/workspaces/langchain/libs/langchain/langchain/tools/slack/base.py`)
```
class SlackBaseTool(BaseTool):
    """Base class for Slack tools."""

    client: WebClient=Field(default_factory=authenticate)    
```

`utils.py`
in (`/workspaces/langchain/libs/langchain/langchain/tools/slack/utils.py`)
```
def authenticate() -> WebClient:
"""Authenticate using the Slack API."""
token = os.environ["SLACK_BOT_TOKEN"]
client = WebClient(token=token)
return client
```

`send_message.py` in (`/workspaces/langchain/libs/langchain/langchain/tools/slack/send_message.py`)
```
class SendMessageSchema(BaseModel):
"""Input for SendMessageTool."""
    body
    to

class SlackSendMessage(SlackBaseTool):
    name: str 
    description: str
    args_schema: Type[SendMessageSchema] = SendMessageSchema

def _run(
        self,
        body: str,
        to: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,) -> str:
        result = self.client.chat_postMessage(
            channel=to,
            text=body
         )
         output = "Message sent: " + str(result)
         return output
```

`get_message.py` in (`/workspaces/langchain/libs/langchain/langchain/tools/slack/get_message.py`)

```
class GetMessagesSchema(BaseModel):
    channel_name: str
    message: str

class SlackGetMessages:
    name = "Get Slack Messages"
    description = "Retrieve messages from a Slack channel by channel name"
    args_schema = Type[GetMessagesSchema] = GetMessagesSchema

    def __init__(self, slack_client):
        self.client = slack_client

    def _run(self, channel_name: str):
        # Initialize conversation_id to None
        conversation_id = None

        # Search for the conversation with the specified channel name
        try:
            # Iterate through the list of channels using the conversations.list method
            for result in self.client.conversations_list():
                if conversation_id is not None:
                    break
                for channel in result["channels"]:
                    if channel["name"] == channel_name:
                        conversation_id = channel["id"]
                        # Print the result
                        print(f"Found conversation ID: {conversation_id}")
                        break

        except SlackApiError as e:
            print(f"Error: {e}")

        # Initialize conversation_history list
        conversation_history = []

        try:
            # Call the conversations.history method using the WebClient
            # conversations.history returns the first 100 messages by default
            result = self.client.conversations_history(channel=conversation_id)

            conversation_history = result["messages"] 

	     # assign the specific message to the "message" field
     if conversation_history:
            	 message = conversation_history[0]["text"]

		
            # Print the results
            print(f"{len(conversation_history)} messages found in {channel_name}")

        except SlackApiError as e:
            print(f"Error retrieving conversation history: {e}")

        # You can return or process the conversation history as needed
        return conversation_history
```

# Mock Examples

`slack.ipynb`

## Slack
```
!pip install slack_sdk
```

## Assign environmental variables
```
```

## Create the Toolkit and Get Tools
```
from langchain.agents.agent_toolkits import SlackToolkit

toolkit = SlackToolkit()
tools = toolkit.get_tools()
tools
```

## Use within an agent
```
from langchain.llms import OpenAI
from langchain.agents import initialize_agent, AgentType
```
```
llm = OpenAI(temperature=0)
agent = initialize_agent(
    tools=toolkit.get_tools(),
    llm=llm,
    verbose=False,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
)
```
```
agent.run(
    "Please send the following message to #general in the CSCD01 workspace: 'Hello world!'"
)
```



# Discussions with Community

1. https://github.com/langchain-ai/langchain/issues/11775
