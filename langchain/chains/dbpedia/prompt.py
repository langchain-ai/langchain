from langchain.chains.prompt_selector import ConditionalPromptSelector, is_chat_model
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import HumanMessage

TEMPLATE = """Write a sparkql query to execute against DBPedia to answer the following question

Question: {question}
SPARQL Query:"""
PROMPT = PromptTemplate.from_template(TEMPLATE)

INSTRUCTIONS_TEMPLATE = """Write a sparkql query to execute against DBPedia to answer the following question.
Your answer should be a valid SPARKQL query and NOTHING else.
Always return just a SPARKQL query."""
INSTRUCTIONS = HumanMessage(content=INSTRUCTIONS_TEMPLATE)
CHAT_PROMPT = ChatPromptTemplate.from_messages(
    [INSTRUCTIONS, HumanMessagePromptTemplate.from_template("{question}")]
)

PROMPT_SELECTOR = ConditionalPromptSelector(
    default_prompt=PROMPT, conditionals=[(is_chat_model, CHAT_PROMPT)]
)

ANSWER_TEMPLATE = """Write a sparkql query to execute against DBPedia to answer the following question

Question: {question}
SPARKQL Query: {query}
SPARKQL Response: {response}
Final Answer (in plain English):"""
ANSWER_PROMPT = PromptTemplate.from_template(ANSWER_TEMPLATE)

ANSWER_INSTRUCTIONS_TEMPLATE = """I wrote this SPARKQL query:
----------
{query}
----------

I got this response:
----------
{response}
----------

Now, use the above information to answer my next question."""
ANSWER_INSTRUCTIONS = HumanMessagePromptTemplate.from_template(
    ANSWER_INSTRUCTIONS_TEMPLATE
)
ANSWER_CHAT_PROMPT = ChatPromptTemplate.from_messages(
    [ANSWER_INSTRUCTIONS, HumanMessagePromptTemplate.from_template("{question}")]
)

ANSWER_PROMPT_SELECTOR = ConditionalPromptSelector(
    default_prompt=ANSWER_PROMPT, conditionals=[(is_chat_model, ANSWER_CHAT_PROMPT)]
)
