import asyncio
import logging
import os
from typing import List, Optional, Type

from firecrawl import FirecrawlApp
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class ScrapeWebsiteInput(BaseModel):
    base_url: str = Field(..., description="The base URL to scrape.")


class FirecrawlScrapeWebsiteTool(BaseTool):
    name: str = "scrape_website"
    description: str = (
        "A tool for scraping websites and extracting markdown content. "
        "Provide the URL of the website to scrape and indicate if the scraping rate "
        "should be limited. The parameters required are:\n"
        "- 'base_url': The base URL of the website to scrape.\n"
        "- 'rate_limit_enabled': A boolean specifying whether to limit the scraping rate."
    )
    args_schema: Type[BaseModel] = ScrapeWebsiteInput
    return_direct: bool = True

    api_key: str = Field(
        default = os.getenv("FIRECRAWL_API_KEY", ""),
        description="API key for Firecrawl",
    )
    rate_limit_enabled: bool = Field(
        default=os.getenv("FIRECRAWL_RATE_LIMIT_ENABLED", "true")
        .strip()
        .lower()
        == "true",
        description="Limit scraping rate",
    )

    app: Optional[FirecrawlApp] = None

    def __init__(
        self, api_key: Optional[str] = None, rate_limit_enabled: Optional[bool] = None
    ):
        super().__init__()
        self.api_key = api_key or self.api_key
        self.rate_limit_enabled = rate_limit_enabled if rate_limit_enabled is not None else self.rate_limit_enabled
        if not self.api_key:
            raise ValueError(
                "API key for Firecrawl is required. Please set the "
                "FIRECRAWL_API_KEY environment variable or pass it to the "
                "constructor."
            )
        self.app = FirecrawlApp(api_key=self.api_key)

    def _run(
        self,
        base_url: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool synchronously."""
        if run_manager:
            run_manager.on_text("Starting web crawl...")
        result = asyncio.run(self.scrape_all_urls(base_url))
        if run_manager:
            run_manager.on_text(f"Completed web crawl. Found {len(result)} URLs.")
        return result

    async def _arun(
        self,
        base_url: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        if run_manager:
            await run_manager.on_text("Starting asynchronous web crawl...")
        result = await self.scrape_all_urls(base_url)
        if run_manager:
            await run_manager.on_text(f"Completed asynchronous web crawl. Found {len(result)} URLs.")
        return result

    async def scrape_all_urls(self, base_url: str) -> str:
        app = self.get_firecrawl_app()

        urls = self.map_website(app, base_url)
        if not urls:
            return "No URLs found. Please check if the base URL is correct."

        markdown_content = ""
        for i, url in enumerate(urls):
            content = await self.async_scrape_url(app, url)
            markdown_content += f"# {url}\n\n{content}\n\n---\n\n"

            # Flexible rate limiting
            if self.rate_limit_enabled and (i + 1) % 10 == 0:
                logging.info("Rate limiting: Sleeping for 60 seconds")
                await asyncio.sleep(60)

        return markdown_content

    def get_firecrawl_app(self) -> FirecrawlApp:
        return self.app

    def map_website(self, app: FirecrawlApp, url: str) -> List[str]:
        try:
            map_status = self.app.map_url(url)
            logging.info(f"Map status: {map_status}")
            if isinstance(map_status, dict) and "links" in map_status:
                return map_status["links"]
            else:
                logging.warning(
                    "Map status is not a list or does not contain 'links': %s",
                    map_status,
                )
                return []
        except Exception as e:
            logging.error(f"Error mapping website {url}: {e}")
            raise RuntimeError(f"Failed to map website: {url}") from e


    async def async_scrape_url(self, app: FirecrawlApp, url: str) -> str:
        try:
            scrape_status = app.scrape_url(url)
            if "markdown" in scrape_status:
                return scrape_status["markdown"]
            else:
                logging.warning("No markdown content found for URL: %s", url)
                return ""
        except Exception as e:
            logging.error(f"Error scraping URL {url}: {e}")
            raise RuntimeError(f"Failed to scrape URL: {url}") from e
