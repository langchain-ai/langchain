import requests
from bs4 import BeautifulSoup
from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.document_loaders.parsers.grobid import GrobidParser

from typing import Any, Iterator, List, Mapping, Optional
from langchain.document_loaders.blob_loaders import Blob

class ServerUnavailableException(Exception):
    pass
  
class GrobidLoader(BaseLoader):
    """Loader that uses Grobid to load article PDF files."""
    def __init__(self,file_path,segment_sentences,grobid_server="http://localhost:8070/api/processFulltextDocument"):
        try:
            from bs4 import BeautifulSoup  # noqa:F401
        except ImportError:
            raise ImportError(
                "`bs4` package not found, please install it with "
                "`pip install bs4`"
            )
        try:
            r = requests.get(grobid_server)
        except:
            print("GROBID server does not appear up and running, please ensure Grobid is installed and the server is running")
            raise ServerUnavailableException
        pdf = open(file_path, "rb")
        files = {"input": (file_path,pdf,"application/pdf",{"Expires": "0"})}
        try:
          data = {}
          for param in ['generateIDs','consolidateHeader','segmentSentences']:data[param]='1'
          data["teiCoordinates"]=["head","s"]
      
          headers = {"Accept":"application/xml"}
          files = files or {}
          r = requests.request(
              "POST",
              grobid_server,
              headers=None,
              params=None,
              files=files,
              data=data,
              timeout=60,
          )
          xml_data,status = r.text,r.status_code
        except requests.exceptions.ReadTimeout:
          status,xml_data=408, None
              
        self.file_path=file_path
        self.xml_data=xml_data
        self.segment_sentences=segment_sentences
              
    def load(self) -> List[Document]:
        """Load file."""
        parser = GrobidParser()
        if self.xml_data==None:
          return None
        return parser.lazy_parse(self.file_path,self.xml_data,self.segment_sentences)
