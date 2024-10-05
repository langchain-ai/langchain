from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from langchain_community.retrievers.sineps import (
    SinepsAttributeInfo,
    SinepsSelfQueryRetriever,
)


def test_sineps_retriever():
    docs = [
        Document(
            page_content="A bunch of scientists bring back"
            "dinosaurs and mayhem breaks loose",
            metadata={"year": 1993, "rating": 7.7, "genre": "science fiction"},
        ),
        Document(
            page_content="Leo DiCaprio gets lost in a dream"
            "within a dream within a dream within a ...",
            metadata={"year": 2010, "director": "Christopher Nolan", "rating": 8.2},
        ),
        Document(
            page_content="A psychologist / detective gets lost in a series"
            "of dreams within dreams within dreams and Inception reused the idea",
            metadata={"year": 2006, "director": "Satoshi Kon", "rating": 8.6},
        ),
        Document(
            page_content="A bunch of normal-sized women are supremely"
            "wholesome and some men pine after them",
            metadata={"year": 2019, "director": "Greta Gerwig", "rating": 8.3},
        ),
        Document(
            page_content="Toys come alive and have a blast doing so",
            metadata={"year": 1995, "genre": "animated"},
        ),
        Document(
            page_content="Three men walk into the Zone, three men walk out of the Zone",
            metadata={
                "year": 1979,
                "director": "Andrei Tarkovsky",
                "genre": "science fiction",
                "rating": 9.9,
            },
        ),
    ]
    vectorstore = Chroma.from_documents(docs, OpenAIEmbeddings())
    sineps_metadata_field_info = [
        SinepsAttributeInfo(
            name="genre",
            description="The genre of the movie.",
            type="string",
            values=[
                "science fiction",
                "comedy",
                "drama",
                "thriller",
                "romance",
                "action",
                "animated",
            ],
        ),
        SinepsAttributeInfo(
            name="year",
            description="The year the movie was released",
            type="number",
        ),
        SinepsAttributeInfo(
            name="director",
            description="The name of the movie director",
            type="string",
            values=[
                "Christopher Nolan",
                "Satoshi Kon",
                "Greta Gerwig",
                "Andrei Tarkovsky",
            ],
        ),
        SinepsAttributeInfo(
            name="rating", description="A 1-10 rating for the movie", type="number"
        ),
    ]
    retriever = SinepsSelfQueryRetriever(
        vectorstore=vectorstore,
        sineps_metadata_field_info=sineps_metadata_field_info,
        verbose=True,
    )
    res = retriever.invoke("I want to watch a movie rated higher than 8.5")

    assert len(res) == 2
