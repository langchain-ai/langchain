import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate

# Load your free Gemini (Makersuite) API key
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Please set your GOOGLE_API_KEY environment variable.")

# ✅ Use free model supported with Makersuite
llm = ChatGoogleGenerativeAI(
    model="models/chat-bison-001",  # ✅ Free model for Makersuite users
    temperature=0.7,
    google_api_key=api_key
)

memory = ConversationBufferMemory()

prompt = PromptTemplate(
    input_variables=["history", "input"],
    template="""
You are a helpful assistant.
{history}
Human: {input}
AI:"""
)

conversation = ConversationChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    verbose=True,
)

print("Start chatting with Gemini (type 'exit' to quit')\n")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    response = conversation.predict(input=user_input)
    print("Gemini:", response)
