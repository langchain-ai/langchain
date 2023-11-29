from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import FAISS

loader = CSVLoader('/Users/harrisonchase/Downloads/titanic.csv')

docs = loader.load()
index_creator = VectorstoreIndexCreator(vectorstore_cls=FAISS)

index = index_creator.from_documents(docs)

index.vectorstore.save_local("titanic_data")
