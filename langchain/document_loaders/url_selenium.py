"""Loader that uses Selenium to load a page, then uses unstructured to load the html.
"""
import logging
from typing import TYPE_CHECKING, List, Literal, Optional, Union

if TYPE_CHECKING:
    from selenium.webdriver import Chrome, Firefox

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

logger = logging.getLogger(__name__)


class SeleniumURLLoader(BaseLoader):
    """Loader that uses Selenium and to load a page and unstructured to load the html.
    This is useful for loading pages that require javascript to render.

    Attributes:
        urls (List[str]): List of URLs to load.
        continue_on_failure (bool): If True, continue loading other URLs on failure.
        browser (str): The browser to use, either 'chrome' or 'firefox'.
        binary_location (Optional[str]): The location of the browser binary.
        executable_path (Optional[str]): The path to the browser executable.
        headless (bool): If True, the browser will run in headless mode.
        arguments [List[str]]: List of arguments to pass to the browser.
    """

    def __init__(
        self,
        urls: List[str],
        continue_on_failure: bool = True,
        browser: Literal["chrome", "firefox"] = "chrome",
        binary_location: Optional[str] = None,
        executable_path: Optional[str] = None,
        headless: bool = True,
        arguments: List[str] = [],
    ):
        """Load a list of URLs using Selenium and unstructured."""
        try:
            import selenium  # noqa:F401
        except ImportError:
            raise ImportError(
                "selenium package not found, please install it with "
                "`pip install selenium`"
            )

        try:
            import unstructured  # noqa:F401
        except ImportError:
            raise ImportError(
                "unstructured package not found, please install it with "
                "`pip install unstructured`"
            )

        self.urls = urls
        self.continue_on_failure = continue_on_failure
        self.browser = browser
        self.binary_location = binary_location
        self.executable_path = executable_path
        self.headless = headless
        self.arguments = arguments

    def _get_driver(self) -> Union["Chrome", "Firefox"]:
        """Create and return a WebDriver instance based on the specified browser.

        Raises:
            ValueError: If an invalid browser is specified.

        Returns:
            Union[Chrome, Firefox]: A WebDriver instance for the specified browser.
        """
        if self.browser.lower() == "chrome":
            from selenium.webdriver import Chrome
            from selenium.webdriver.chrome.options import Options as ChromeOptions

            chrome_options = ChromeOptions()

            for arg in self.arguments:
                chrome_options.add_argument(arg)

            if self.headless:
                chrome_options.add_argument("--headless")
                chrome_options.add_argument("--no-sandbox")
            if self.binary_location is not None:
                chrome_options.binary_location = self.binary_location
            if self.executable_path is None:
                return Chrome(options=chrome_options)
            return Chrome(executable_path=self.executable_path, options=chrome_options)
        elif self.browser.lower() == "firefox":
            from selenium.webdriver import Firefox
            from selenium.webdriver.firefox.options import Options as FirefoxOptions

            firefox_options = FirefoxOptions()

            for arg in self.arguments:
                firefox_options.add_argument(arg)

            if self.headless:
                firefox_options.add_argument("--headless")
            if self.binary_location is not None:
                firefox_options.binary_location = self.binary_location
            if self.executable_path is None:
                return Firefox(options=firefox_options)
            return Firefox(
                executable_path=self.executable_path, options=firefox_options
            )
        else:
            raise ValueError("Invalid browser specified. Use 'chrome' or 'firefox'.")

    def load(self) -> List[Document]:
        """Load the specified URLs using Selenium and create Document instances.

        Returns:
            List[Document]: A list of Document instances with loaded content.
        """
        from unstructured.partition.html import partition_html

        docs: List[Document] = list()
        driver = self._get_driver()

        for url in self.urls:
            try:
                driver.get(url)
                page_content = driver.page_source
                elements = partition_html(text=page_content)
                text = "\n\n".join([str(el) for el in elements])
                metadata = {"source": url}
                docs.append(Document(page_content=text, metadata=metadata))
            except Exception as e:
                if self.continue_on_failure:
                    logger.error(f"Error fetching or processing {url}, exception: {e}")
                else:
                    raise e

        driver.quit()
        return docs
