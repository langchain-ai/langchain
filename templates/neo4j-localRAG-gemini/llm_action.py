from langchain_google_genai import ChatGoogleGenerativeAI

class LLMAction:
    def __init__(self, api_key, model="gemini-pro"):
        self.api_key = api_key
        self.model = model
        self.llm = ChatGoogleGenerativeAI(model=model, google_api_key=api_key)

    def generate_cypher_query(self, user_prompt):
        result = self.llm.invoke("Write a cypher query where I need you to " + user_prompt)
        cypher_query = result.content.split("```cypher")[1].split("```")[0].strip()
        return cypher_query
