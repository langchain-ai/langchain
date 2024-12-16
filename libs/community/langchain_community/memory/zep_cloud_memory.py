from __future__ import annotations

from typing import Any, Dict, Optional

from langchain_community.chat_message_histories import ZepCloudChatMessageHistory

try:
    from langchain.memory import ConversationBufferMemory
    from zep_cloud import MemoryGetRequestMemoryType

    class ZepCloudMemory(ConversationBufferMemory):  # type: ignore[override]
        """Persist your chain history to the Zep MemoryStore.

        Documentation: https://help.getzep.com

        Example:
            .. code-block:: python

            memory = ZepCloudMemory(
                        session_id=session_id,  # Identifies your user or a user's session
                        api_key=<your_api_key>, # Your Zep Project API key
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


        Zep - Recall, understand, and extract data from chat histories. Power personalized AI experiences.
        =========
        Zep is a long-term memory service for AI Assistant apps. With Zep, you can provide AI assistants with the ability to recall past conversations,
        no matter how distant, while also reducing hallucinations, latency, and cost.

        For more information on the zep-python package, see:
        https://github.com/getzep/zep-python

        """  # noqa: E501

        chat_memory: ZepCloudChatMessageHistory

        def __init__(
            self,
            session_id: str,
            api_key: str,
            memory_type: Optional[MemoryGetRequestMemoryType] = None,
            lastn: Optional[int] = None,
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
                api_key (str): Your Zep Project key.
                memory_type (Optional[MemoryGetRequestMemoryType], optional): Zep Memory Type, defaults to perpetual
                lastn (Optional[int], optional): Number of messages to retrieve. Will add the last summary generated prior to the nth oldest message. Defaults to 6
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
            """  # noqa: E501
            chat_message_history = ZepCloudChatMessageHistory(
                session_id=session_id,
                memory_type=memory_type,
                lastn=lastn,
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
except ImportError:
    # Placeholder object
    class ZepCloudMemory:  # type: ignore[no-redef]
        pass
