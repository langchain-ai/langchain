from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.schema import Document

# Here is a description of the dataset and metadata attributes. Metadata attributes will
# be used to filter the results of the query beyond the semantic search.
DEFAULT_DOCUMENT_CONTENTS = "Brief summary of a movie"
DEFAULT_METADATA_FIELD_INFO = [
    AttributeInfo(
        name="genre",
        description="The genre of the movie",
        type="string or list[string]",
    ),
    AttributeInfo(
        name="year",
        description="The year the movie was released",
        type="integer",
    ),
    AttributeInfo(
        name="director",
        description="The name of the movie director",
        type="string",
    ),
    AttributeInfo(
        name="rating", description="A 1-10 rating for the movie", type="float"
    ),
]

# A default set of documents to use for the Vectara platform. This is a list of Document
# objects, which have a page_content field and a metadata field. The metadata field is a
# dictionary of metadata attributes compatible with the metadata field info above.
DEFAULT_DOCUMENTS = [
    Document(
        page_content='''
            A bunch of scientists bring back dinosaurs and mayhem breaks loose
        ''',
        metadata={"year": 1993, "rating": 7.7, "genre": "science fiction"},
    ),
    Document(
        page_content='''
        Leo DiCaprio gets lost in a dream within a dream within a dream within a ...
        ''',
        metadata={"year": 2010, "director": "Christopher Nolan", "rating": 8.2},
    ),
    Document(
        page_content='''
        A psychologist / detective gets lost in a series of dreams within dreams 
        within dreams and Inception reused the idea
        ''',
        metadata={"year": 2006, "director": "Satoshi Kon", "rating": 8.6},
    ),
    Document(
        page_content='''
        A bunch of normal-sized women are supremely wholesome and some men 
        pine after them
        ''',
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
            "rating": 9.9,
            "director": "Andrei Tarkovsky",
            "genre": "science fiction",
        },
    ),
]
