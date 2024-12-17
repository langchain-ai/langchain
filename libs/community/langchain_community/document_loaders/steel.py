"""Steel Web Document Loader for LangChain.

This module provides a web document loader using Steel's browser automation 
and Playwright for web page content extraction.

Key Features:
- Load web pages from single or multiple URLs
- Support for Steel's proxy and CAPTCHA solving
- Optional screenshot capture
- Robust error handling
"""
from __future__ import annotations

import logging
import os
from typing import List, Optional, Union

import urllib3
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader

from steel import Steel
from playwright.sync_api import sync_playwright, TimeoutError

# Configure logging to reduce noise from external libraries
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('playwright').setLevel(logging.WARNING)

class SteelWebLoader(BaseLoader):
    """Loader that uses Steel and Playwright to load web pages.

    This loader supports advanced web page loading with:
    - Single or multiple URL loading
    - Optional screenshot capture
    - Proxy and CAPTCHA solving via Steel
    
    Attributes:
        urls (List[str]): URLs to load
        steel_api_key (Optional[str]): Authentication key for Steel
        timeout (int): Navigation timeout in milliseconds
        use_proxy (bool): Use Steel's proxy network
        solve_captcha (bool): Enable CAPTCHA solving
        take_screenshot (bool): Capture page screenshots
    """

    def __init__(
        self,
        urls: Union[str, List[str]],
        steel_api_key: Optional[str] = None,
        timeout: int = 30000,
        use_proxy: bool = True,
        solve_captcha: bool = False,
        take_screenshot: bool = False,
        log_level: int = logging.INFO
    ):
        """
        Initialize the Steel Web Loader.

        Args:
            urls: URL or list of URLs to load
            steel_api_key: Steel API key (optional if set in environment)
            timeout: Navigation timeout in milliseconds
            use_proxy: Use Steel's proxy network
            solve_captcha: Enable CAPTCHA solving
            take_screenshot: Capture page screenshots
            log_level: Logging level for the loader
        
        Raises:
            ValueError: If no Steel API key is provided
        """
        # Normalize URLs
        self.urls = [urls] if isinstance(urls, str) else urls
        
        # API Key from environment or parameter
        self.steel_api_key = steel_api_key or os.getenv('STEEL_API_KEY')
        if not self.steel_api_key:
            raise ValueError(
                "Steel API key is required. "
                "Set STEEL_API_KEY environment variable or provide steel_api_key."
            )
        
        self.timeout = timeout
        self.use_proxy = use_proxy
        self.solve_captcha = solve_captcha
        self.take_screenshot = take_screenshot
        
        # Configure logging
        logging.basicConfig(level=log_level, 
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def load(self) -> List[Document]:
        """
        Load documents from web pages using Steel and Playwright.

        Returns:
            List of Documents extracted from web pages
        """
        documents = []
        
        for url in self.urls:
            session = None
            
            try:
                # Create Steel client and session
                client = Steel(steel_api_key=self.steel_api_key)
                session = client.sessions.create(
                    use_proxy=self.use_proxy,
                    solve_captcha=self.solve_captcha
                )
                
                # Playwright connection
                with sync_playwright() as playwright:
                    browser = playwright.chromium.connect_over_cdp(
                        f"wss://connect.steel.dev?apiKey={self.steel_api_key}&sessionId={session.id}"
                    )
                    
                    context = browser.contexts[0]
                    page = context.new_page()
                    
                    # Navigate to URL with enhanced error handling
                    try:
                        response = page.goto(url, wait_until="networkidle", timeout=self.timeout)
                        
                        # Additional check for HTTP errors
                        if response and response.status >= 400:
                            self.logger.warning(f"HTTP error {response.status} for {url}")
                            continue
                    
                    except TimeoutError:
                        self.logger.warning(f"Timeout loading {url}")
                        continue
                    except Exception as e:
                        self.logger.error(f"Navigation error for {url}: {e}")
                        continue
                    
                    # Extract content with fallback
                    try:
                        content = page.inner_text('body') or page.content()
                    except Exception as e:
                        self.logger.error(f"Content extraction error for {url}: {e}")
                        content = f"Content extraction failed: {str(e)}"
                    
                    # Optional screenshot with error tracking
                    screenshot_path = None
                    if self.take_screenshot:
                        try:
                            screenshots_dir = os.path.join(os.getcwd(), 'screenshots')
                            os.makedirs(screenshots_dir, exist_ok=True)
                            screenshot_path = os.path.join(
                                screenshots_dir, 
                                f"{url.replace('https://', '').replace('/', '_')}_screenshot.png"
                            )
                            page.screenshot(path=screenshot_path, full_page=True)
                        except Exception as e:
                            self.logger.error(f"Screenshot error for {url}: {e}")
                    
                    # Create document
                    doc = Document(
                        page_content=content,
                        metadata={
                            'source': url,
                            'steel_session_id': session.id,
                            'screenshot_path': screenshot_path
                        }
                    )
                    documents.append(doc)
            
            except Exception as e:
                self.logger.error(f"Critical error processing {url}: {e}")
            
            finally:
                # Always release the session
                if session:
                    try:
                        client.sessions.release(session.id)
                    except Exception as e:
                        self.logger.error(f"Error releasing session: {e}")
        
        return documents