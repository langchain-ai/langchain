from __future__ import annotations

from typing import Any, Dict, Optional

from langchain_community.chat_message_histories import ZepChatMessageHistory

from langchain.memory import ConversationBufferMemory


class ZepMemory(ConversationBufferMemory):
    """Persist your chain history to the Zep MemoryStore.

    The number of messages returned by Zep and when the Zep server summarizes chat
    histories is configurable. See the Zep documentation for more details.

    Documentation: https://docs.getzep.com

    Example:
        .. code-block:: python

        memory = ZepMemory(
                    session_id=session_id,  # Identifies your user or a user's session
                    url=ZEP_API_URL,        # Your Zep server's URL
                    api_key=<your_api_key>, # Optional
                    memory_key="history",   # Ensure this matches the key used in
                                            # chain's prompt template
                    return_messages=True,   # Does your prompt template expect a string
                                            # or a list of Messages?
                )
        chain = LLMChain(memory=memory,...) # Configure your chain to use the ZepMemory
                                              instance


    Note:
        To persist metadata alongside your chat history, your will need to create a
    custom Chain class that overrides the `prep_outputs` method to include the metadata
    in the call to `self.memory.save_context`.


    Zep - Fast, scalable building blocks for LLM Apps
    =========
    Zep is an open source platform for productionizing LLM apps. Go from a prototype
    built in LangChain or LlamaIndex, or a custom app, to production in minutes without
    rewriting code.

    For server installation instructions and more, see:
    https://docs.getzep.com/deployment/quickstart/

    For more information on the zep-python package, see:
    https://github.com/getzep/zep-python

    """

    chat_memory: ZepChatMessageHistory

    def __init__(
        self,
        session_id: str,
        url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        output_key: Optional[str] = None,
        input_key: Optional[str] = None,
        return_messages: bool = False,
        human_prefix: str = "Human",
        ai_prefix: str = "AI",
        memory_key: str = "history",
    ):
        """Initialize ZepMemory.

        Args:
            session_id (str): Identifies your user or a user's session
            url (str, optional): Your Zep server's URL. Defaults to
                                 "http://localhost:8000".
            api_key (Optional[str], optional): Your Zep API key. Defaults to None.
            output_key (Optional[str], optional): The key to use for the output message.
                                              Defaults to None.
            input_key (Optional[str], optional): The key to use for the input message.
                                              Defaults to None.
            return_messages (bool, optional): Does your prompt template expect a string
                                              or a list of Messages? Defaults to False
                                              i.e. return a string.
            human_prefix (str, optional): The prefix to use for human messages.
                                          Defaults to "Human".
            ai_prefix (str, optional): The prefix to use for AI messages.
                                       Defaults to "AI".
            memory_key (str, optional): The key to use for the memory.
                                        Defaults to "history".
                                        Ensure that this matches the key used in
                                        chain's prompt template.
        """
        chat_message_history = ZepChatMessageHistory(
            session_id=session_id,
            url=url,
            api_key=api_key,
        )
        super().__init__(
            chat_memory=chat_message_history,
            output_key=output_key,
            input_key=input_key,
            return_messages=return_messages,
            human_prefix=human_prefix,
            ai_prefix=ai_prefix,
            memory_key=memory_key,
        )

    def save_context(
        self,
        inputs: Dict[str, Any],
        outputs: Dict[str, str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save context from this conversation to buffer.

        Args:
            inputs (Dict[str, Any]): The inputs to the chain.
            outputs (Dict[str, str]): The outputs from the chain.
            metadata (Optional[Dict[str, Any]], optional): Any metadata to save with
                                                           the context. Defaults to None

        Returns:
            None
        """
        input_str, output_str = self._get_input_output(inputs, outputs)
        self.chat_memory.add_user_message(input_str, metadata=metadata)
        self.chat_memory.add_ai_message(output_str, metadata=metadata)
