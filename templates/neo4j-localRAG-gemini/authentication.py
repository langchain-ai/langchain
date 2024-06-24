class Authentication:
    def __init__(self):
        self.graph_db_url = None
        self.password = None
        self.google_api_key = None

    def authenticate(self, graph_db_url, password, google_api_key):
        self.graph_db_url = graph_db_url
        self.password = password
        self.google_api_key = google_api_key
        return self.graph_db_url and self.password and self.google_api_key
