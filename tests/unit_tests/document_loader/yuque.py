from langchain.document_loaders import YUQUELoader


loader = YUQUELoader()

doc = loader.load()
print(doc)
