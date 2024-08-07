from langchain_snowflake.chat import ChatSnowflakeCortex

model = ChatSnowflakeCortex(model="mistral-7b")

print(model.invoke("Hello!"))
