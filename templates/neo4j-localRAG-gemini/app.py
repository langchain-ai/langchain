from neo4j_database import Neo4jDatabase
from authentication import Authentication
from llm_action import LLMAction
import config


class App:
    def __init__(self):
        self.auth = Authentication()
        self.db = None
        self.llm_service = None

    def authenticate(self):
        if self.auth.authenticate(
            config.GRAPH_DB_URL, config.GRAPH_DB_PASSWORD, config.GEMINI_API_KEY
        ):
            self.db = Neo4jDatabase(config.GRAPH_DB_URL, config.GRAPH_DB_PASSWORD)
            self.db.init_driver()
            self.llm_action = LLMAction(config.GEMINI_API_KEY)
            print("Authentication successful!")
        else:
            print("Authentication failed. Please provide all required inputs.")

    def handle_prompt(self, prompt):
        if prompt:
            cypher_query = self.llm_action.generate_cypher_query(prompt)
            try:
                results = self.db.run_query(cypher_query)
                return cypher_query, results
            except Exception as e:
                print(f"Error executing the query: {e}")
                return None, None
        else:
            print("Please enter a prompt or query")
            return None, None

    def close(self):
        if self.db:
            self.db.close()
