"""Implement an LLM driven browser."""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.language_models import BaseLanguageModel
from langchain_core.pydantic_v1 import Extra, root_validator

from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chains.natbot.prompt import PROMPT


class NatBotChain(Chain):
    """Implement an LLM driven browser.

    **Security Note**: This toolkit provides code to control a web-browser.

        The web-browser can be used to navigate to:

        - Any URL (including any internal network URLs)
        - And local files

        Exercise care if exposing this chain to end-users. Control who is able to
        access and use this chain, and isolate the network access of the server
        that hosts this chain.

        See https://python.langchain.com/docs/security for more information.

    Example:
        .. code-block:: python

            from langchain.chains import NatBotChain
            natbot = NatBotChain.from_default("Buy me a new hat.")
    """

    llm_chain: LLMChain
    objective: str
    """Objective that NatBot is tasked with completing."""
    llm: Optional[BaseLanguageModel] = None
    """[Deprecated] LLM wrapper to use."""
    input_url_key: str = "url"  #: :meta private:
    input_browser_content_key: str = "browser_content"  #: :meta private:
    previous_command: str = ""  #: :meta private:
    output_key: str = "command"  #: :meta private:

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @root_validator(pre=True)
    def raise_deprecation(cls, values: Dict) -> Dict:
        if "llm" in values:
            warnings.warn(
                "Directly instantiating an NatBotChain with an llm is deprecated. "
                "Please instantiate with llm_chain argument or using the from_llm "
                "class method."
            )
            if "llm_chain" not in values and values["llm"] is not None:
                values["llm_chain"] = LLMChain(llm=values["llm"], prompt=PROMPT)
        return values

    @classmethod
    def from_default(cls, objective: str, **kwargs: Any) -> NatBotChain:
        """Load with default LLMChain."""
        raise NotImplementedError(
            "This method is no longer implemented. Please use from_llm."
            "llm = OpenAI(temperature=0.5, best_of=10, n=3, max_tokens=50)"
            "For example, NatBotChain.from_llm(llm, objective)"
        )

    @classmethod
    def from_llm(
        cls, llm: BaseLanguageModel, objective: str, **kwargs: Any
    ) -> NatBotChain:
        """Load from LLM."""
        llm_chain = LLMChain(llm=llm, prompt=PROMPT)
        return cls(llm_chain=llm_chain, objective=objective, **kwargs)

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

    def _call(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        url = inputs[self.input_url_key]
        browser_content = inputs[self.input_browser_content_key]
        llm_cmd = self.llm_chain.predict(
            objective=self.objective,
            url=url[:100],
            previous_command=self.previous_command,
            browser_content=browser_content[:4500],
            callbacks=_run_manager.get_child(),
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

    @property
    def _chain_type(self) -> str:
        return "nat_bot_chain"
