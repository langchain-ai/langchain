# Copyright (c) 2023 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main authors: Kevin Monsalvez-Pozo and Jorge Ruiz

from typing import Dict, List, Any

from .abstract_language_model import AbstractLanguageModel

from langchain_openai import ChatOpenAI



class GPTmodels(AbstractLanguageModel):
    """
    The Gemini class handles interactions with the Gemini models using the provided configuration.

    Inherits from the AbstractLanguageModel and implements its abstract methods.
    """

    def __init__(
        self, config_path: str = "", model_name: str = "", cache: bool = False, api_key: str = ""
    ) -> None:
        """
        Initialize the Gemini instance with configuration, model details, and caching options.

        :param config_path: Path to the configuration file. Defaults to "".
        :type config_path: str
        :param model_name: Name of the model, default is 'gemini'. Used to select the correct configuration.
        :type model_name: str
        :param cache: Flag to determine whether to cache responses. Defaults to False.
        :type cache: bool
        """
        super().__init__(config_path, model_name, cache)
        self.config: Dict = self.config[model_name]
        # The model_id is the id of the model that is used to generate responses.
        self.model_id: str = self.config["model_id"]
        # The temperature of a model is defined as the randomness of the model's output.
        self.temperature: float = self.config["temperature"]
        # The maximum number of tokens to generate in the chat completion.
        self.max_tokens: int = self.config["max_tokens"]
        self.api_key: str = api_key


    def load_llm(self, ) -> Any:
        """
        Abstract method to load the language model.

        :return: The language model instance.
        :rtype: Any
        """

        llm = ChatOpenAI(model_name = self.model_id, 
                        max_tokens= self.max_tokens, 
                        temperature = self.temperature,
                        api_key = self.api_key
                        )

        return llm
   