from langchain.llms import Replicate
from langchain.prompts import ChatPromptTemplate

# LLM
replicate_id = "andreasjansson/llama-2-13b-chat-gguf:60ec5dda9ff9ee0b6f786c9d1157842e6ab3cc931139ad98fe99e08a35c5d4d4"  # noqa: E501
model = Replicate(
    model=replicate_id,
    model_kwargs={"temperature": 0.8, "max_length": 500, "top_p": 0.95},
)

# Prompt with output schema specification
template = """A article will be passed to you. Extract from it all papers that are mentioned by this article. 

Do not extract the name of the article itself. If no papers are mentioned that's fine - you don't need to extract any! Just return an empty list.

Do not make up or guess ANY extra information. Only extract what exactly is in the text.

Respond with json that adheres to the following jsonschema:

{{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {{
    "author": {{
      "type": "string",
      "description": "The author of the paper."
    }},
    "title": {{
      "type": "string",
      "description": "The title of the paper."
    }}
  }},
  "required": ["author", "title"],
  "additionalProperties": false
}}"""  # noqa: E501

prompt = ChatPromptTemplate.from_messages([("system", template), ("human", "{input}")])

# Chain
chain = prompt | model
