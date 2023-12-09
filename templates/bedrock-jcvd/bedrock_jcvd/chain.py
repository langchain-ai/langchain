from langchain.chat_models import BedrockChat
from langchain.prompts import ChatPromptTemplate

_model = BedrockChat(
    model_id="anthropic.claude-v2",
    model_kwargs={
        "temperature": 0.1
    }
)

_prompt = ChatPromptTemplate.from_messages([
    ("human", "You are JCVD. {input}"),
])

chain = _prompt | _model