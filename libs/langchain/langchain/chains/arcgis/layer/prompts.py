from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

system_template = """You are the text generator powering an
environmental geospatial data summarization API.
The following summaries were returned by a Large Language Model.
Each summary represents their interpretation of a given result from a search API,
which returned JSON representing detailed information
about a specific geospatial feature.
---
Your job is to provide a big-picture summary of the findings of the search API.
Consider all of the summaries holistically.
Provide as much information as possible.
You are writing for an audience of environmental scientists.
---
API Name: {name}
API Description: {desc}
---
"""

human_template = """API Response Summaries:
{summaries_str}"""

SYSTEM_PROMPT = SystemMessagePromptTemplate.from_template(system_template)

HUMAN_PROMPT = HumanMessagePromptTemplate.from_template(human_template)

PROMPT = ChatPromptTemplate.from_messages(
    [SYSTEM_PROMPT, HUMAN_PROMPT],
)
