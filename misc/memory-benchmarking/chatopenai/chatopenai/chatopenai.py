from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "What is the capital of {country}?"),
    ]
)

chat = ChatOpenAI()

chain = prompt | chat

chain.invoke({"country": "Denmark"})
