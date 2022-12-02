import weaviate

client = weaviate.Client(
	url="https://testing-open-harrison.semi.network",
	additional_headers={
        'X-OpenAI-Api-Key': 
    }
)
content = {
    "concepts": ["What did the president say about Ketanji Brown Jackson"],
}
result = client.query.get("Paragraph", ["content", "permalink"]).with_near_text(content).with_limit(4).do()
print(result)
