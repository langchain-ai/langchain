"""Chain that calls Google Books API."""
import requests

from typing import Any, Dict, Optional

from langchain_core.utils import get_from_dict_or_env
from pydantic import BaseModel, ConfigDict, model_validator



class GoogleBooksAPIWrapper(BaseModel):
    gbooks_api_key: Optional[str] = None
    top_k_results: int = 3

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """Validate that api key is in your environment variable."""
        gbooks_api_key = get_from_dict_or_env(
            values, "gbooks_api_key", "GBOOKS_API_KEY"
        )
        values["gbooks_api_key"] = gbooks_api_key
    
    def run(self, query: str) -> str:
        """"""
        url = f'https://www.googleapis.com/books/v1/volumes'
    
        # Set up query parameters
        params = {
            'q': query,
            'key': self.gbooks_api_key,
            'maxResults': self.top_k_results,
        }
        
        response = requests.get(url, params=params)
        
        # If the request was successful, return the books
        if response.status_code == 200:
            books = response.json().get('items', [])
            formatted_books = []
            for book in books:
                title = book['volumeInfo'].get('title', 'No title available')
                authors = book['volumeInfo'].get('authors', 'No authors available')
                description = book['volumeInfo'].get('description', 'No description available')

                formatted_book = f'"{title}" by {", ".join(authors) if isinstance(authors, list) else authors}: {description}'
                formatted_books.append(formatted_book)
            
            return formatted_books
        else:
            print(f"Failed to retrieve books: {response.status_code}")
            return []
