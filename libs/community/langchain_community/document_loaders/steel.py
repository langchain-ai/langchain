"""Load web pages using Steel.dev browser automation.

Example:
    .. code-block:: python

        from langchain_community.document_loaders import SteelWebLoader
        
        loader = SteelWebLoader(
            "https://example.com",
            steel_api_key="your-api-key",
            extract_strategy="text"
        )
        documents = loader.load()
"""
from typing import List, Optional, Dict, Any
import logging
import asyncio
import os
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
from playwright.async_api import async_playwright

class SteelWebLoader(BaseLoader):
    """Load web pages using Steel.dev browser automation.

    This loader uses Steel.dev's managed browser infrastructure to load web pages,
    with support for proxy networks and automated CAPTCHA solving.
    """
    
    def __init__(
        self, 
        url: str, 
        steel_api_key: Optional[str] = None,
        extract_strategy: str = 'text',
        timeout: int = 30000,
        use_proxy: bool = True,
        solve_captcha: bool = True
    ) -> None:
        """Initialize the Steel Web Loader.

        Args:
            url: Web page URL to load
            steel_api_key: Steel API key. If not provided, will look for STEEL_API_KEY env var
            extract_strategy: Content extraction method ('text', 'markdown', or 'html')
            timeout: Navigation timeout in milliseconds
            use_proxy: Whether to use Steel's proxy network
            solve_captcha: Whether to enable automated CAPTCHA solving
        
        Raises:
            ValueError: If extract_strategy is invalid or no API key is provided
        """
        self.url = url
        self.steel_api_key = steel_api_key or os.getenv('STEEL_API_KEY')
        if not self.steel_api_key:
            raise ValueError(
                "steel_api_key must be provided or STEEL_API_KEY environment variable must be set"
            )
        
        self.extract_strategy = extract_strategy
        self.timeout = timeout
        self.use_proxy = use_proxy
        self.solve_captcha = solve_captcha
        
        self.logger = logging.getLogger(__name__)
        
        valid_strategies = ['text', 'markdown', 'html']
        if extract_strategy not in valid_strategies:
            raise ValueError(
                f"Invalid extract_strategy. Must be one of {valid_strategies}"
            )
    
    async def _create_session(self) -> Dict[str, Any]:
        """Create a new Steel session.
        
        Returns:
            Dict containing session information including ID and viewer URL
        """
        # Initialize Playwright
        playwright = await async_playwright().start()
        
        try:
            # Create session with Steel
            params = []
            if self.use_proxy:
                params.append("useProxy=true")
            if self.solve_captcha:
                params.append("solveCaptcha=true")
            
            params_str = "&".join(params)
            connection_url = f"wss://connect.steel.dev?apiKey={self.steel_api_key}"
            if params_str:
                connection_url += f"&{params_str}"
            
            # Connect to Steel
            browser = await playwright.chromium.connect_over_cdp(connection_url)
            
            # Get session ID from connection URL
            session_id = connection_url.split("sessionId=")[-1].split("&")[0]
            
            return {
                "id": session_id,
                "viewer_url": f"https://app.steel.dev/sessions/{session_id}",
                "browser": browser,
                "playwright": playwright
            }
            
        except Exception as e:
            await playwright.stop()
            raise e
    
    async def _aload(self) -> List[Document]:
        """Async implementation of web page loading.

        Returns:
            List[Document]: List containing the loaded web page as a Document

        Raises:
            Exception: If page loading fails
        """
        try:
            # Create session
            session = await self._create_session()
            self.logger.info(f"Created Steel session: {session['id']}")
            
            try:
                # Create new page
                context = session["browser"].contexts[0]
                page = await context.new_page()
                
                # Navigate to URL
                await page.goto(
                    self.url, 
                    wait_until="networkidle", 
                    timeout=self.timeout
                )
                
                # Extract content based on strategy
                if self.extract_strategy == 'text':
                    content = await page.inner_text('body')
                elif self.extract_strategy == 'markdown':
                    content = await page.inner_text('body')  # TODO: Implement markdown conversion
                else:  # html
                    content = await page.content()
                
                # Create document
                return [
                    Document(
                        page_content=content,
                        metadata={
                            'source': self.url,
                            'steel_session_id': session['id'],
                            'steel_session_viewer_url': session['viewer_url'],
                            'extract_strategy': self.extract_strategy
                        }
                    )
                ]
            
            finally:
                # Always close the browser and stop Playwright
                await session["browser"].close()
                await session["playwright"].stop()
        
        except Exception as e:
            self.logger.error(f"Error loading {self.url}: {e}")
            raise
    
    def load(self) -> List[Document]:
        """Load the web page.

        Returns:
            List[Document]: List containing the loaded web page as a Document
            
        Raises:
            Exception: If page loading fails
        """
        return asyncio.run(self._aload())
