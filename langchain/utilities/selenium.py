"""Util that calls Selenium."""

import re
import time
from typing import Optional

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By


class SeleniumWrapper:
    """Wrapper around Selenium.

    To use, you should have the ``selenium`` python package installed.

    Example:
        .. code-block:: python

            from langchain import SeleniumWrapper
            selenium = SeleniumWrapper()
    """

    def __init__(self) -> None:
        """Initialize Selenium and start interactive session."""
        chrome_options = Options()
        chrome_options.add_argument("--start-maximized")
        self.driver = webdriver.Chrome(options=chrome_options)

    def __del__(self) -> None:
        """Close Selenium session."""
        self.driver.close()

    def describe_website(self, url: Optional[str] = None) -> str:
        """Describe the website."""
        if url:
            try:
                self.driver.get(url)
            except Exception:
                return f"Cannot load website {url}."
        # Extract headings
        heading_tags = ["h1", "h2", "h3", "h4", "h5", "h6"]
        heading_output = "Heading Output: "
        for tag in heading_tags:
            headings = self.driver.find_elements(By.XPATH, f"//{tag}")
            for heading in headings:
                heading_text = heading.text.strip()
                if heading_text:
                    heading_output += f"{tag.upper()}: {heading_text}  "

        # Extract paragraphs
        paragraphs = self.driver.find_elements(By.XPATH, "//p")
        paragraph_output = "Paragraph: "
        for paragraph in paragraphs:
            paragraph_text = re.sub(r"\s+", " ", paragraph.text.strip())
            if paragraph_text:
                paragraph_output += f"{paragraph_text}  "

        # Extract interactable components (buttons and links)
        buttons = self.driver.find_elements(By.XPATH, "//button")
        links = self.driver.find_elements(By.XPATH, "//a")

        interactable_output = "Interactable Elements: "
        for element in buttons + links:
            element_tag = element.tag_name
            element_text = element.text.strip()
            element_location = element.location
            if element_text and "\n" not in element_text:
                interactable_output += (
                    f"Interactable Element: {element_tag}, "
                    f"Text: {element_text}, "
                    f"Location: {element_location}\n"
                )

        return f"{heading_output}\n\n{paragraph_output}\n\n{interactable_output}"

    def click_button_by_text(self, element_text: str) -> str:
        """Click a button element with text."""
        try:
            # Find both buttons and links
            xpath = (
                "//button[translate(normalize-space(.//text()), "
                "'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz')='"
                f"{element_text.lower()}'] | //a[translate(normalize-space(.//text()), "
                "'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz')='"
                f"{element_text.lower()}']"
            )
            elements = self.driver.find_elements(By.XPATH, xpath)

            if not elements:
                return f"No interactable element found with text: {element_text}"

            element = elements[0]

            # Scroll the element into view
            self.driver.execute_script("arguments[0].scrollIntoView();", element)
            time.sleep(1)  # Allow some time for the page to settle

            # Click the element using JavaScript
            self.driver.execute_script("arguments[0].click();", element)
            output = f"Clicked interactable element with text: {element_text}\n"
            return output
        except Exception as e:
            return (
                f"Error clicking interactable element with text '{element_text}': {e}"
            )
