from langchain_core.prompts import PromptTemplate

template = """You are a teacher coming up with questions to ask on a quiz.
Given the following document, please generate a question and answer based on that document.

Example Format:
<Begin Document>
...
<End Document>
QUESTION: question here
ANSWER: answer here

These questions should be detailed and be based explicitly on information in the document. Begin!

<Begin Document>
{doc}
<End Document>"""  # noqa: E501
PROMPT = PromptTemplate(
    input_variables=["doc"],
    template=template,
)
