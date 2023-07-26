import os
import sys

# Get the current directory path
current_directory = os.path.dirname(os.path.abspath(__file__))
print(current_directory)
# Add the current directory to sys.path
sys.path.insert(0, current_directory)

from langchain.embeddings.awa import AwaEmbeddings

Embedding = AwaEmbeddings()

print(Embedding.embed_query("The test information"))
print(Embedding.embed_documents(["test1", "another test"]))