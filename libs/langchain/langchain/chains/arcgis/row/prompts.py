from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

system_template = """You are the text generator powering an
environmental geospatial data summarization API.
You respond only in declarative, non-interactive bullet-points.
You will be presented with JSON representing detailed information
about a specific geospatial feature.
You will return a natural language, bullet point summary of the feature.
Provide as much information as possible.
You are writing for an audience of environmental scientists.
---
API Name: {name}
API Description: {desc}
"""

human_template = """API Data: ```json
{json_str}
```
"""

SYSTEM_PROMPT = SystemMessagePromptTemplate.from_template(
    system_template,
)

HUMAN_PROMPT = HumanMessagePromptTemplate.from_template(
    human_template,
)

PROMPT = ChatPromptTemplate.from_messages(
    [SYSTEM_PROMPT, HUMAN_PROMPT],
)
