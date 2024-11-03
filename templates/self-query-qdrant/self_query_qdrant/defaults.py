from langchain.chains.query_constructor.schema import AttributeInfo
from langchain_core.documents import Document

# Qdrant collection name
DEFAULT_COLLECTION_NAME = "restaurants"

# Here is a description of the dataset and metadata attributes. Metadata attributes will
# be used to filter the results of the query beyond the semantic search.
DEFAULT_DOCUMENT_CONTENTS = (
    "Dishes served at different restaurants, along with the restaurant information"
)
DEFAULT_METADATA_FIELD_INFO = [
    AttributeInfo(
        name="price",
        description="The price of the dish",
        type="float",
    ),
    AttributeInfo(
        name="restaurant.name",
        description="The name of the restaurant",
        type="string",
    ),
    AttributeInfo(
        name="restaurant.location",
        description="Name of the city where the restaurant is located",
        type="string or list[string]",
    ),
]

# A default set of documents to use for the vector store. This is a list of Document
# objects, which have a page_content field and a metadata field. The metadata field is a
# dictionary of metadata attributes compatible with the metadata field info above.
DEFAULT_DOCUMENTS = [
    Document(
        page_content="Pepperoni pizza with extra cheese, crispy crust",
        metadata={
            "price": 10.99,
            "restaurant": {
                "name": "Pizza Hut",
                "location": ["New York", "Chicago"],
            },
        },
    ),
    Document(
        page_content="Spaghetti with meatballs and tomato sauce",
        metadata={
            "price": 12.99,
            "restaurant": {
                "name": "Olive Garden",
                "location": ["New York", "Chicago", "Los Angeles"],
            },
        },
    ),
    Document(
        page_content="Chicken tikka masala with naan",
        metadata={
            "price": 14.99,
            "restaurant": {
                "name": "Indian Oven",
                "location": ["New York", "Los Angeles"],
            },
        },
    ),
    Document(
        page_content="Chicken teriyaki with rice",
        metadata={
            "price": 11.99,
            "restaurant": {
                "name": "Sakura",
                "location": ["New York", "Chicago", "Los Angeles"],
            },
        },
    ),
    Document(
        page_content="Scabbard fish with banana and passion fruit sauce",
        metadata={
            "price": 19.99,
            "restaurant": {
                "name": "A Concha",
                "location": ["San Francisco"],
            },
        },
    ),
    Document(
        page_content="Pielmieni with sour cream",
        metadata={
            "price": 13.99,
            "restaurant": {
                "name": "Russian House",
                "location": ["New York", "Chicago"],
            },
        },
    ),
    Document(
        page_content="Chicken biryani with raita",
        metadata={
            "price": 14.99,
            "restaurant": {
                "name": "Indian Oven",
                "location": ["Los Angeles"],
            },
        },
    ),
    Document(
        page_content="Tomato soup with croutons",
        metadata={
            "price": 7.99,
            "restaurant": {
                "name": "Olive Garden",
                "location": ["New York", "Chicago", "Los Angeles"],
            },
        },
    ),
    Document(
        page_content="Vegan burger with sweet potato fries",
        metadata={
            "price": 12.99,
            "restaurant": {
                "name": "Burger King",
                "location": ["New York", "Los Angeles"],
            },
        },
    ),
    Document(
        page_content="Chicken nuggets with french fries",
        metadata={
            "price": 9.99,
            "restaurant": {
                "name": "McDonald's",
                "location": ["San Francisco", "New York", "Los Angeles"],
            },
        },
    ),
]
