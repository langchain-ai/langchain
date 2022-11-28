# flake8: noqa
from langchain.prompts import PromptTemplate

question_prompt_template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Accurate Answer:"""
question_prompt = PromptTemplate(
    template=question_prompt_template, input_variables=["context", "question"]
)

combine_prompt_template = """Given the following questions, reference links and associated content, create a final answer with references. If ANY of the content contains relevant information, please use that even if all the others do not.

Question: What color is an apple?
=========
Content: An apple can be red
Source: foo
Content: An apple can be green
Source: bar
Content: An orange is orange
Source: baz
=========
Final Answer: An apple can be red or green
Sources: foo, bar

Question: What color is an apple?
=========
Content: An orange is orange
Source: foo
Content: A pinapple is yellow
Source: bar
Content: A plum is purple
Source: baz
Content: An apple is green
Source: foobar
=========
Final Answer: green
Sources: foobar

Question: What color is an apple?
=========
Content: An orange is orange
Source: foo
Content: A pinapple is yellow
Source: foo
Content: A plum is purple
Source: foo
=========
Final Answer: It is unclear from the given info
Sources:

Question: {question}
=========
{summaries}
=========
Final Answer:"""
combine_prompt = PromptTemplate(
    template=combine_prompt_template, input_variables=["summaries", "question"]
)

example_prompt = PromptTemplate(
    template="Content: {page_content}\nSource: {source}",
    input_variables=["page_content", "source"],
)
