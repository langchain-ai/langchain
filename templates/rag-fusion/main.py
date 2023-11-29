from rag_fusion.chain import chain

if __name__ == "__main__":
	original_query = "impact of climate change"
	print(chain.invoke(original_query))
