"""Test Vertex AI API wrapper.
In order to run this test, you need to install VertexAI SDK and Pillow
pip install google-cloud-aiplatform>=1.35.0
pip install pillow

"""

from langchain_experimental.vertex_ai.vertex_ai import VertexAIMultimodalEmbeddings

def test_embedding_image() -> None:
    documents = ["tests/examples/espresso_part.png"]
    model = VertexAIMultimodalEmbeddings()
    output = model.embed_images(documents)
    assert len(output) == 1
    assert len(output[0]) == 1408

def test_embedding_text() -> None:
    documents = ["test123"]
    model = VertexAIMultimodalEmbeddings()
    output = model.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) == 1408

def test_embed_query() -> None:
    document = ["test123"]
    model = VertexAIMultimodalEmbeddings()
    output = model.embed_documents(document)
    print(output)
    assert len(output) == 1
    assert len(output[0]) == 1408
