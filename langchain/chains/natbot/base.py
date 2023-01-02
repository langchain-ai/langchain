"""Implement an LLM driven browser."""
from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, Extra

from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chains.natbot.prompt import PROMPT
from langchain.llms.base import BaseLLM
from langchain.llms.openai import OpenAI


class NatBotChain(Chain, BaseModel):
    """Implement an LLM driven browser.

    Example:
        .. code-block:: python

            from langchain import NatBotChain, OpenAI
            natbot = NatBotChain(llm=OpenAI(), objective="Buy me a new hat.")
    """

    llm: BaseLLM
    """LLM wrapper to use."""
    objective: str
    """Objective that NatBot is tasked with completing."""
    input_url_key: str = "url"  #: :meta private:
    input_browser_content_key: str = "browser_content"  #: :meta private:
    previous_command: str = ""  #: :meta private:
    output_key: str = "command"  #: :meta private:

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @classmethod
    def from_default(cls, objective: str) -> NatBotChain:
        """Load with default LLM."""
        llm = OpenAI(temperature=0.5, best_of=10, n=3, max_tokens=50)
        return cls(llm=llm, objective=objective)

    @property
    def input_keys(self) -> List[str]:
        """Expect url and browser content.

        :meta private:
        """
        return [self.input_url_key, self.input_browser_content_key]

    @property
    def output_keys(self) -> List[str]:
        """Return command.

        :meta private:
        """
        return [self.output_key]

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        llm_executor = LLMChain(prompt=PROMPT, llm=self.llm)
        url = inputs[self.input_url_key]
        browser_content = inputs[self.input_browser_content_key]
        llm_cmd = llm_executor.predict(
            objective=self.objective,
            url=url[:100],
            previous_command=self.previous_command,
            browser_content=browser_content[:4500],
        )
        llm_cmd = llm_cmd.strip()
        self.previous_command = llm_cmd
        return {self.output_key: llm_cmd}

    def execute(self, url: str, browser_content: str) -> str:
        """Figure out next browser command to run.

        Args:
            url: URL of the site currently on.
            browser_content: Content of the page as currently displayed by the browser.

        Returns:
            Next browser command to run.

        Example:
            .. code-block:: python

                browser_content = "...."
                llm_command = natbot.run("www.google.com", browser_content)
        """
        _inputs = {
            self.input_url_key: url,
            self.input_browser_content_key: browser_content,
        }
        return self(_inputs)[self.output_key]
