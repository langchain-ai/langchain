import logging
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class DriaAPIWrapper:
    """Wrapper around Dria API.

    This wrapper facilitates interactions with Dria's vector search
    and retrieval services, including creating knowledge bases, inserting data,
    and fetching search results.

    Attributes:
        api_key: Your API key for accessing Dria.
        contract_id: The contract ID of the knowledge base to interact with.
        top_n: Number of top results to fetch for a search.
    """

    def __init__(
        self, api_key: str, contract_id: Optional[str] = None, top_n: int = 10
    ):
        try:
            from dria import Dria, Models
        except ImportError:
            logger.error(
                """Dria is not installed. Please install Dria to use this wrapper.
                
                You can install Dria using the following command:
                pip install dria
                """
            )
            return

        self.api_key = api_key
        self.models = Models
        self.contract_id = contract_id
        self.top_n = top_n
        self.dria_client = Dria(api_key=self.api_key)
        if self.contract_id:
            self.dria_client.set_contract(self.contract_id)

    def create_knowledge_base(
        self,
        name: str,
        description: str,
        category: str,
        embedding: str,
    ) -> str:
        """Create a new knowledge base."""
        contract_id = self.dria_client.create(
            name=name, embedding=embedding, category=category, description=description
        )
        logger.info(f"Knowledge base created with ID: {contract_id}")
        self.contract_id = contract_id
        return contract_id

    def insert_data(self, data: List[Dict[str, Any]]) -> str:
        """Insert data into the knowledge base."""
        response = self.dria_client.insert_text(data)
        logger.info(f"Data inserted: {response}")
        return response

    def search(self, query: str) -> List[Dict[str, Any]]:
        """Perform a text-based search."""
        results = self.dria_client.search(query, top_n=self.top_n)
        logger.info(f"Search results: {results}")
        return results

    def query_with_vector(self, vector: List[float]) -> List[Dict[str, Any]]:
        """Perform a vector-based query."""
        vector_query_results = self.dria_client.query(vector, top_n=self.top_n)
        logger.info(f"Vector query results: {vector_query_results}")
        return vector_query_results

    def run(self, query: Union[str, List[float]]) -> Optional[List[Dict[str, Any]]]:
        """Method to handle both text-based searches and vector-based queries.

        Args:
            query: A string for text-based search or a list of floats for
            vector-based query.

        Returns:
            The search or query results from Dria.
        """
        if isinstance(query, str):
            return self.search(query)
        elif isinstance(query, list) and all(isinstance(item, float) for item in query):
            return self.query_with_vector(query)
        else:
            logger.error(
                """Invalid query type. Please provide a string for text search or a 
                list of floats for vector query."""
            )
            return None
