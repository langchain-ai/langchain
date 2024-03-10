import csv

from langchain.chains.question_answering import load_qa_chain
from langchain_community.embeddings import CohereEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate

from .chat import chat

csv_file = open("data/books_with_blurbs.csv", "r")
csv_reader = csv.reader(csv_file)
csv_data = list(csv_reader)
parsed_data = [
    {
        "id": x[0],
        "title": x[1],
        "author": x[2],
        "year": x[3],
        "publisher": x[4],
        "blurb": x[5],
    }
    for x in csv_data
]
parsed_data[1]

embeddings = CohereEmbeddings()

docsearch = Chroma.from_texts(
    [x["title"] for x in parsed_data], embeddings, metadatas=parsed_data
).as_retriever()


prompt_template = """
{context}

Use the book reccommendations to suggest books for the user to read.
Only use the titles of the books, do not make up titles. Format the response as
a bulleted list prefixed by a relevant message.

User: {message}"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "message"]
)

book_rec_chain = {
    "input_documents": lambda x: docsearch.get_relevant_documents(x["message"]),
    "message": lambda x: x["message"],
} | load_qa_chain(chat, chain_type="stuff", prompt=PROMPT)
