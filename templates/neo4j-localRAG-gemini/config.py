# config.py
import os
from dotenv import load_dotenv

load_dotenv()

GRAPH_DB_URL = os.getenv("Neo4j_url")
GRAPH_DB_PASSWORD = os.getenv("Neo4j_password")
GEMINI_API_KEY = os.getenv("Gemini_api_key")
