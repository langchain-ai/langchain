"""Test FireCrawlLoader."""
import pytest
from unittest.mock import MagicMock, patch

from langchain_community.document_loaders.firecrawl import FireCrawlLoader


class TestFireCrawlLoader:
    """Test FireCrawlLoader."""

    @patch("firecrawl.FireCrawl")
    def test_load_extract_mode(self, mock_firecrawl_class):
        """Test loading in extract mode."""
        # Setup mock
        mock_client = MagicMock()
        mock_firecrawl_class.return_value = mock_client

        response_dict = {
            'success': True,
            'data': {
                'title': 'extracted title',
                'main contents': 'extracted main contents'
            },
            'status': 'completed',
            'expiresAt': '2025-03-12T12:42:09.000Z'
        }
        mock_client.extract.return_value = response_dict
        
        params = {
            "prompt": "extract the title and main contents(write your own prompt here)",
            "schema": {              
            "type": "object",         
            "properties": {           
                "title": {"type": "string"},                      
                "main contents": {"type": "string"}                       
            },                        
            "required": [             
                "title",                     
                "main contents"              
            ]                              
            },                               
            "enableWebSearch": False,      
            "ignoreSitemap": False,           
            "showSources": False,          
            "scrapeOptions": {             
                "formats": [                 
                "markdown"                 
                ],                           
                "onlyMainContent": True,     
                "headers": {},               
                "waitFor": 0,                
                "mobile": False,             
                "skipTlsVerification": False,
                "timeout": 30000,            
                "removeBase64Images": True,  
                "blockAds": True,     
                "proxy": "basic"
                }
        }
        
        loader = FireCrawlLoader(
            url="https://example.com",
            api_key="fake-key",
            mode="extract",
            params=params
        )
        docs = loader.load()
        
        # 검증
        assert len(docs) == 1
        

        
