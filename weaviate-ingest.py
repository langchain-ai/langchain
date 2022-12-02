import weaviate

client = weaviate.Client(
    url="https://testing-open-harrison.semi.network",
    additional_headers={
        'X-OpenAI-Api-Key': 
    }
)
client.schema.delete_all()
client.schema.get()
schema = {
    "classes": [
        {
            "class": "Paragraph",
            "description": "A written comment, by an author",
            "vectorizer": "text2vec-openai",
              "moduleConfig": {
                "text2vec-openai": {
                  "model": "babbage",
                  "type": "text"
                }
              },
            "properties": [
                {
                    "dataType": ["text"],
                    "description": "The content of the paragraph",
                    "moduleConfig": {
                        "text2vec-openai": {
                          "skip": False,
                          "vectorizePropertyName": False
                        }
                      },
                    "name": "content",
                },
                {
                    "dataType": ["text"],
                    "description": "Permalink to the comment.",
                    "name": "permalink",
                },
            ],
        },
    ]
}

client.schema.create(schema)

from langchain.text_splitter import CharacterTextSplitter

with open("docs/examples/state_of_the_union.txt") as f:
    state_of_the_union = f.read()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_text(state_of_the_union)

docs = [{"content": t, "permalink": f"Paragraph-{i}"} for i, t in enumerate(texts)]

with client.batch as batch:
    for comment in docs:
        batch.add_data_object(comment, "Paragraph")
