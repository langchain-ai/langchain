from bs4 import BeautifulSoup
from langchain_community.document_loaders import WebBaseLoader

class SSearX:
    def __init__(self):
        pass

    # Function to load website content using WebBaseLoader
    def load_website_content(self, url):
        """
        Loads content from a specified URL using the WebBaseLoader class.
        
        Args:
            url (str): The website URL to load content from.

        Returns:
            str or None: Returns the loaded content if successful, or None if an error occurs.
        """
        try:
            loader = WebBaseLoader(url)
            return loader.load()
        except Exception as e:
            print(f"Error: {e}")
            return None

    def ssxGoogle(self, query):
        """
        Performs a Google search and retrieves content from the search results page.

        Args:
            query (str): The search phrase to query on Google.

        Returns:
            str or None: Returns the loaded search results page content if successful, or None if an error occurs.
        """
        # Format query for Google search URL
        search_query = query.replace(" ", "+")
        url = f"https://www.google.com/search?q={search_query}"

        try:
            # Load the Google search results page content
            website_data = self.load_website_content(url)
            return website_data
        except Exception as e:
            print("No information was captured.")
            return None

    def ssxWebsite(self, url):
        """
        Loads and retrieves content from a specified website.

        Args:
            url (str): The base URL (without 'https://www.') of the website to access.

        Returns:
            str: The loaded website content if successful, or a message indicating that no information was captured.
        """
        # Format URL with HTTPS and 'www'
        url_ = f"https://www.{url}"
        
        try:
            # Load the specified website content
            website_data = self.load_website_content(url_)
            return website_data
        except Exception as e:
            print("Result: No information was captured.")
            return None

# Example usage:
# ssearx_instance = SSearX()
# print(ssearx_instance.ssxWebsite("example.com"))
