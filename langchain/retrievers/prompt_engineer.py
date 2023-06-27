from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

class PromptEngineerRetriever:

    """ Use an LLM to produce a specified number of altnerative prompts for retrival. Query the retriever with all prompts and return the unique results."""
    
    # Prompt 
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question", "num_queries"],
        template="""You are helping a user perform retrieval of documents from a vector database. 

        Take the user question, and create {num_queries} queries that capture that modify the wording in order to maximize the likelihood that relevant documents are retrieved.

        Query: {question}

        Alterative queries with each seperated by a newline:"""
    )

    def __init__(self, retriever, num_queries):
        """ Simply pass a retriever and a specified number of queries """
        self.retriever = retriever
        self.num_queries = num_queries
        self.llm = self.init_llm()

    def init_llm(self):
        return ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    def get_relevant_documents(self, question):
        queries = self.generate_queries(question)
        documents = self.retrieve_documents(queries)
        unique_documents = self.unique_union(documents)
        return unique_documents

    def generate_queries(self, question):
        prompt = self.QUERY_PROMPT.format(question=question, num_queries=self.num_queries)
        response = self.llm.predict(prompt)
        queries = response.split("\n")
        return queries

    def retrieve_documents(self, queries):
        documents = []
        for query in queries:
            docs = self.retriever.get_relevant_documents(query)  # assuming this method exists
            documents.extend(docs)
        return documents

    def unique_union(self, documents):
        # Create a dictionary with page_content as keys. This will automatically remove duplicates
        unique_documents_dict = {doc.page_content: doc for doc in documents}
        # Return the unique documents as a list
        unique_documents = list(unique_documents_dict.values())
        return unique_documents