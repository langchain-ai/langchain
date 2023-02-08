"""Loader that loads webpages"""
from typing import List

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


class WebpageLoader(BaseLoader):
    """Loader that uses urllib and beautiful soup to load webpages."""

    def __init__(self, file_path: str):
        """Initialize with webpage path."""
        self.file_path = file_path

    """Custom BeautifulSoup parser methods. TODO: Add new parsing logic below. """
    def imsdb(self, soup): 
        transcript = soup.select_one("td[class='scrtext']").text
        return transcript
        
    def azlyrics(self, soup): 
        title = soup.title.text
        lyrics = soup.find_all("div", {"class": ""})[2].text
        return title + lyrics
    
    def college_confidential(self, soup):
        text = soup.select_one("main[class='skin-handler']").text
        return text


    def load(self) -> List[Document]:
        """Load webpage."""
        from urllib.request import urlopen

        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ValueError(
                "Could not import bs4 python package. "
                "Please it install it with `pip install beautifulsoup4`."
            )
        

        html_doc = urlopen(self.file_path)
        soup = BeautifulSoup(html_doc, 'html.parser')

        text = ""
        if self.file_path.startswith("https://www.azlyrics.com/lyrics/"):
            text = self.azlyrics(soup)
        elif self.file_path.startswith("https://imsdb.com/scripts/"):
            text = self.imsdb(soup)
        elif self.file_path.startswith("https://www.collegeconfidential.com/colleges/"):
            text = self.college_confidential(soup)
        else: 
            raise ValueError('URL prefix is not supported. See documentation for how to add new bs4 parser.')


        metadata = {"source": self.file_path}
        return [Document(page_content=text, metadata=metadata)]
