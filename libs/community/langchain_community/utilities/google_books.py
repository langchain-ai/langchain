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
        return values

    def run(self, query: str) -> str:
        """Query Google Books API and return a formatted string of book details."""
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
            formatted_books = [f"Here are a few books related to {query}:\n"]
            for i, book in enumerate(books, start=1):
                title = book['volumeInfo'].get('title', 'No title available')
                authors = book['volumeInfo'].get('authors', 'No authors available')
                description = book['volumeInfo'].get('description', 'No description available')

                # Format each book's details
                formatted_book = f'{i}. "{title}" by {", ".join(authors) if isinstance(authors, list) else authors}: {description}'
                formatted_books.append(formatted_book)
            
            return "\n\n".join(formatted_books)
        else:
            return f"Failed to retrieve books: {response.status_code}"