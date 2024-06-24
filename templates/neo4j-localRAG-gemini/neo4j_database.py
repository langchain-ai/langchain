from neo4j import GraphDatabase


class Neo4jDatabase:
    def __init__(self, url, password):
        self.url = url
        self.password = password
        self.driver = None

    def init_driver(self):
        self.driver = GraphDatabase.driver(self.url, auth=("neo4j", self.password))
        self.driver.verify_connectivity()
        return self.driver

    def run_query(self, query):
        with self.driver.session() as session:
            results = session.run(query)
            return list(results)

    def close(self):
        if self.driver:
            self.driver.close()
