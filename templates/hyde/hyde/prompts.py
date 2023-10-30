from langchain.prompts.prompt import PromptTemplate

# There are a few different templates to choose from
# These are just different ways to generate hypothetical documents
web_search_template = """Please write a passage to answer the question 
Question: {question}
Passage:"""
sci_fact_template = """Please write a scientific paper passage to support/refute the claim 
Claim: {question}
Passage:"""  # noqa: E501
fiqa_template = """Please write a financial article passage to answer the question
Question: {question}
Passage:"""
trec_news_template = """Please write a news passage about the topic.
Topic: {question}
Passage:"""

# For the sake of this example we will use the web search template
hyde_prompt = PromptTemplate.from_template(web_search_template)
