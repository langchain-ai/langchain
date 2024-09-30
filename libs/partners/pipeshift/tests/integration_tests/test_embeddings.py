# """Test Pipeshift embeddings."""

# from langchain_pipeshift import PipeshiftEmbeddings


# def test_langchain_pipeshift_embed_documents() -> None:
#     """Test Pipeshift embeddings."""
#     documents = ["foo bar", "bar foo"]
#     embedding = PipeshiftEmbeddings()
#     output = embedding.embed_documents(documents)
#     assert len(output) == 2
#     assert len(output[0]) > 0


# def test_langchain_pipeshift_embed_query() -> None:
#     """Test Pipeshift embeddings."""
#     query = "foo bar"
#     embedding = PipeshiftEmbeddings()
#     output = embedding.embed_query(query)
#     assert len(output) > 0


# async def test_langchain_pipeshift_aembed_documents() -> None:
#     """Test Pipeshift embeddings asynchronous."""
#     documents = ["foo bar", "bar foo"]
#     embedding = PipeshiftEmbeddings()
#     output = await embedding.aembed_documents(documents)
#     assert len(output) == 2
#     assert len(output[0]) > 0


# async def test_langchain_pipeshift_aembed_query() -> None:
#     """Test Pipeshift embeddings asynchronous."""
#     query = "foo bar"
#     embedding = PipeshiftEmbeddings()
#     output = await embedding.aembed_query(query)
#     assert len(output) > 0
