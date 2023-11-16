from langchain.llms import Replicate
from langchain.prompts import ChatPromptTemplate

# LLM
replicate_id = "andreasjansson/llama-2-13b-chat-gguf:60ec5dda9ff9ee0b6f786c9d1157842e6ab3cc931139ad98fe99e08a35c5d4d4"  # noqa: E501
model = Replicate(
    model=replicate_id,
    model_kwargs={"temperature": 0.8, "max_length": 500, "top_p": 0.95},
)

# Prompt with output schema specification
template = """You are an AI language model assistant. Your task is to generate 3 different versions of the given user /
    question to retrieve relevant documents from a vector  database. By generating multiple perspectives on the user / 
    question, your goal is to help the user overcome some of the limitations  of distance-based similarity search. /
    Respond with json that adheres to the following jsonschema:
{{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {{
    "question_1": {{
      "type": "string",
      "description": "First version of the user question."
    }},
    "question_2": {{
      "type": "string",
      "description": "Second version of the user question."
    }},
    "question_3": {{
      "type": "string",
      "description": "Third version of the user question."
    }}
  }},
  "required": ["question_1","question_2","question_3"],
  "additionalProperties": false
}}"""  # noqa: E501

prompt = ChatPromptTemplate.from_messages(
    [("system", template), ("human", "{question}")]
)

# Chain
chain = prompt | model
