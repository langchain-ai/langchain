"""Load web pages using Steel.dev browser automation."""
import os
from typing import List, Optional
import logging
import asyncio
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader

try:
    from steel import Steel
    from playwright.async_api import async_playwright
except ImportError:
    raise ValueError(
        "Could not import steel-browser-python package. "
        "Please install it with `pip install steel-browser-python`."
    )

logger = logging.getLogger(__name__)

class SteelWebLoader(BaseLoader):
    """Load web pages using Steel.dev browser automation.

    This loader uses Steel.dev's managed browser infrastructure to load web pages,
    with support for proxy networks and automated CAPTCHA solving.

    Example:
        .. code-block:: python

            from langchain_community.document_loaders import SteelWebLoader

            loader = SteelWebLoader(
                "https://example.com",
                steel_api_key="your-api-key"
            )
            documents = loader.load()

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
            ValueError: If extract_strategy is invalid or STEEL_API_KEY is not set
        """
        self.url = url
        self.steel_api_key = steel_api_key or os.getenv("STEEL_API_KEY")
        if not self.steel_api_key:
            raise ValueError(
                "Steel API key must be provided either through steel_api_key parameter "
                "or STEEL_API_KEY environment variable"
            )
        
        self.extract_strategy = extract_strategy
        self.timeout = timeout
        self.use_proxy = use_proxy
        self.solve_captcha = solve_captcha
        
        valid_strategies = ['text', 'markdown', 'html']
        if extract_strategy not in valid_strategies:
            raise ValueError(
                f"Invalid extract_strategy. Must be one of {valid_strategies}"
            )
    
    async def _aload(self) -> List[Document]:
        """Async implementation of web page loading.

        Returns:
            List[Document]: List containing the loaded web page as a Document

        Raises:
            Exception: If page loading fails
        """
        # Initialize Steel client
        client = Steel(steel_api_key=self.steel_api_key)
        
        try:
            # Create Steel session
            session = client.sessions.create(
                use_proxy=self.use_proxy,
                solve_captcha=self.solve_captcha
            )
            logger.info(f"Created Steel session: {session.id}")
            
            # Initialize Playwright
            playwright = await async_playwright().start()
            
            try:
                # Connect to Steel session
                browser = await playwright.chromium.connect_over_cdp(
                    f"wss://connect.steel.dev?apiKey={self.steel_api_key}&sessionId={session.id}"
                )
                
                # Create new page
                context = browser.contexts[0]
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
                    content = await page.inner_text('body')  # Simplified
                else:  # html
                    content = await page.content()
                
                # Create and return document
                return [
                    Document(
                        page_content=content,
                        metadata={
                            'source': self.url,
                            'steel_session_id': session.id,
                            'steel_session_viewer_url': session.session_viewer_url,
                            'extract_strategy': self.extract_strategy
                        }
                    )
                ]
            
            finally:
                await playwright.stop()
                
        except Exception as e:
            logger.error(f"Error loading {self.url}: {e}")
            return []
            
        finally:
            # Always release the session
            try:
                client.sessions.release(session.id)
                logger.info(f"Released Steel session: {session.id}")
            except Exception as e:
                logger.error(f"Error releasing session: {e}")
    
    def load(self) -> List[Document]:
        """Load the web page.

        Returns:
            List[Document]: List containing the loaded web page as a Document
        """
        return asyncio.run(self._aload())
