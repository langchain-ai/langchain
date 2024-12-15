from typing import List, Dict, Any, Optional
from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents import Document
import justext
import requests
import logging
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

class JustextWebLoader(BaseLoader):
    """Loader that uses justext to extract cleaned text from web pages.
    
    This loader removes boilerplate content and navigation elements while preserving
    the main content of web pages. It supports multiple languages and can either combine
    all content into one document or split it into multiple documents by paragraph.
    """
    
    DEFAULT_LANGUAGE = "English"
    
    def __init__(
        self,
        web_path: str,
        language: str = DEFAULT_LANGUAGE,
        split_by_justext_paragraphs: bool = False,
        headers: Optional[Dict[str, str]] = None,
        verify_ssl: bool = True,
        **kwargs: Dict[str, Any]
    ) -> None:
        """Initialize the JustextWebLoader.
        
        Args:
            web_path: URL of the webpage to load
            language: Language for stopwords (default: English)
            split_by_justext_paragraphs: If True, creates separate documents for each paragraph
            headers: Custom headers for requests
            verify_ssl: Whether to verify SSL certificates
            **kwargs: Additional parameters to pass to justext
        """
        self.web_path = web_path
        self.language = language
        self.split_by_justext_paragraphs = split_by_justext_paragraphs
        self.headers = headers or {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        self.verify_ssl = verify_ssl
        self.kwargs = kwargs

    def _extract_metadata(self, html_content: bytes, url: str) -> dict:
        """Extract metadata from the HTML content using BeautifulSoup."""
        soup = BeautifulSoup(html_content, 'html.parser')
        metadata = {
            "source": url,
            "language": self.language
        }
        
        # Extract title
        if title := soup.find("title"):
            metadata["title"] = title.get_text().strip()
        else:
            metadata["title"] = ""
            
        # Extract description
        if description := soup.find("meta", attrs={"name": "description"}):
            metadata["description"] = description.get("content", "No description found.").strip()
        else:
            metadata["description"] = "No description found."
            
        # Extract HTML language attribute
        if html := soup.find("html"):
            metadata["html_language"] = html.get("lang", "No language found.").strip()
        else:
            metadata["html_language"] = "No language found."
            
        return metadata

    def _create_split_documents(self, paragraphs: List[Any], base_metadata: Dict[str, Any]) -> List[Document]:
        """Create separate documents for each paragraph, preserving metadata."""
        docs = []
        for paragraph in paragraphs:
            metadata = base_metadata.copy()
            metadata.update({
                "class_type": paragraph.class_type,
                "heading": paragraph.heading,
                "headings": [paragraph.heading],  # Add headings list for consistency
                "paragraph_count": 1  # Each split document contains one paragraph
            })
            docs.append(Document(
                page_content=paragraph.text.strip(),
                metadata=metadata
            ))
        return docs

    def load(self) -> List[Document]:
        """Load and process the webpage."""
        try:
            # Download webpage
            response = requests.get(
                self.web_path,
                headers=self.headers,
                verify=self.verify_ssl
            )
            response.raise_for_status()
            html = response.content

            # Extract metadata first
            metadata = self._extract_metadata(html, self.web_path)
            
            # Get stopwords for language 
            stoplist = justext.get_stoplist(self.language)
            
            # Extract cleaned paragraphs
            paragraphs = justext.justext(
                html,
                stoplist,
                **self.kwargs
            )

            # Filter content - include both good and near-good for minimal content
            good_paragraphs = [p for p in paragraphs 
                             if not p.is_boilerplate or p.class_type == 'near-good']
            
            # If still no content, try to get any meaningful text
            if not good_paragraphs:
                good_paragraphs = [p for p in paragraphs if p.text.strip()]

            # Return empty list if absolutely no content found
            if not good_paragraphs:
                return []

            if self.split_by_justext_paragraphs:
                return self._create_split_documents(good_paragraphs, metadata)
            return self._create_combined_document(good_paragraphs, metadata)
            
        except requests.RequestException as e:
            logger.error(f"Error fetching {self.web_path}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error processing {self.web_path}: {str(e)}")
            raise
    def _create_combined_document(self, paragraphs: List[Any], base_metadata: Dict[str, Any]) -> List[Document]:
        """Combine paragraphs into a single document with appropriate metadata."""
        content_parts = []
        headings = []
        
        # Collect headings and content
        for paragraph in paragraphs:
            text = paragraph.text.strip()
            if text:  # Only include non-empty paragraphs
                content_parts.append(text)
                if paragraph.heading:
                    headings.append(True)
                else:
                    headings.append(False)

        # Only create document if there's content
        if not content_parts:
            return []
            
        metadata = base_metadata.copy()
        metadata.update({
            "headings": headings,
            "paragraph_count": len(content_parts),
            "class_type": "good"  # All non-boilerplate content is considered good
        })

        return [Document(
            page_content="\n\n".join(content_parts),
            metadata=metadata
        )]