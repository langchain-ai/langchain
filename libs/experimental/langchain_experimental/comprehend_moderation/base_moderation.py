import uuid
from typing import Any, Callable, Optional

from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.prompts.base import StringPromptValue
from langchain.prompts.chat import ChatPromptValue
from langchain.schema import AIMessage, HumanMessage

from langchain_experimental.comprehend_moderation.intent import ComprehendIntent
from langchain_experimental.comprehend_moderation.pii import ComprehendPII
from langchain_experimental.comprehend_moderation.toxicity import ComprehendToxicity


class BaseModeration:
    def __init__(
        self,
        client: Any,
        config: Optional[Any] = None,
        moderation_callback: Optional[Any] = None,
        unique_id: Optional[str] = None,
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ):
        self.client = client
        self.config = config
        self.moderation_callback = moderation_callback
        self.unique_id = unique_id
        self.chat_message_index = 0
        self.run_manager = run_manager
        self.chain_id = str(uuid.uuid4())

    def _convert_prompt_to_text(self, prompt: Any) -> str:
        input_text = str()

        if isinstance(prompt, StringPromptValue):
            input_text = prompt.text
        elif isinstance(prompt, str):
            input_text = prompt
        elif isinstance(prompt, ChatPromptValue):
            """
            We will just check the last message in the message Chain of a
            ChatPromptTemplate. The typical chronology is
            SystemMessage > HumanMessage > AIMessage and so on. However assuming
            that with every chat the chain is invoked we will only check the last
            message. This is assuming that all previous messages have been checked
            already. Only HumanMessage and AIMessage will be checked. We can perhaps
            loop through and take advantage of the additional_kwargs property in the
            HumanMessage and AIMessage schema to mark messages that have been moderated.
            However that means that this class could generate multiple text chunks
            and moderate() logics would need to be updated. This also means some
            complexity in re-constructing the prompt while keeping the messages in
            sequence.
            """
            message = prompt.messages[-1]
            self.chat_message_index = len(prompt.messages) - 1
            if isinstance(message, HumanMessage):
                input_text = message.content

            if isinstance(message, AIMessage):
                input_text = message.content
        else:
            raise ValueError(
                f"Invalid input type {type(input)}. "
                "Must be a PromptValue, str, or list of BaseMessages."
            )
        return input_text

    def _convert_text_to_prompt(self, prompt: Any, text: str) -> Any:
        if isinstance(prompt, StringPromptValue):
            return StringPromptValue(text=text)
        elif isinstance(prompt, str):
            return text
        elif isinstance(prompt, ChatPromptValue):
            # Copy the messages because we may need to mutate them.
            # We don't want to mutate data we don't own.
            messages = list(prompt.messages)

            message = messages[self.chat_message_index]

            if isinstance(message, HumanMessage):
                messages[self.chat_message_index] = HumanMessage(
                    content=text,
                    example=message.example,
                    additional_kwargs=message.additional_kwargs,
                )
            if isinstance(message, AIMessage):
                messages[self.chat_message_index] = AIMessage(
                    content=text,
                    example=message.example,
                    additional_kwargs=message.additional_kwargs,
                )
            return ChatPromptValue(messages=messages)
        else:
            raise ValueError(
                f"Invalid input type {type(input)}. "
                "Must be a PromptValue, str, or list of BaseMessages."
            )

    def _moderation_class(self, moderation_class: Any) -> Callable:
        return moderation_class(
            client=self.client,
            callback=self.moderation_callback,
            unique_id=self.unique_id,
            chain_id=self.chain_id,
        ).validate

    def _log_message_for_verbose(self, message: str) -> None:
        if self.run_manager:
            self.run_manager.on_text(message)

    def moderate(self, prompt: Any) -> str:
        from langchain_experimental.comprehend_moderation.base_moderation_config import (  # noqa: E501
            ModerationIntentConfig,
            ModerationPiiConfig,
            ModerationToxicityConfig,
        )
        from langchain_experimental.comprehend_moderation.base_moderation_exceptions import (  # noqa: E501
            ModerationIntentionError,
            ModerationPiiError,
            ModerationToxicityError,
        )

        try:
            # convert prompt to text
            input_text = self._convert_prompt_to_text(prompt=prompt)
            output_text = str()

            # perform moderation
            filter_functions = {
                "pii": ComprehendPII,
                "toxicity": ComprehendToxicity,
                "intent": ComprehendIntent,
            }

            filters = self.config.filters  # type: ignore

            for _filter in filters:
                filter_name = (
                    "pii"
                    if isinstance(_filter, ModerationPiiConfig)
                    else (
                        "toxicity"
                        if isinstance(_filter, ModerationToxicityConfig)
                        else (
                            "intent"
                            if isinstance(_filter, ModerationIntentConfig)
                            else None
                        )
                    )
                )
                if filter_name in filter_functions:
                    self._log_message_for_verbose(
                        f"Running {filter_name} Validation...\n"
                    )
                    validation_fn = self._moderation_class(
                        moderation_class=filter_functions[filter_name]
                    )
                    input_text = input_text if not output_text else output_text
                    output_text = validation_fn(
                        prompt_value=input_text,
                        config=_filter.dict(),
                    )

            # convert text to prompt and return
            return self._convert_text_to_prompt(prompt=prompt, text=output_text)

        except ModerationPiiError as e:
            self._log_message_for_verbose(f"Found PII content..stopping..\n{str(e)}\n")
            raise e
        except ModerationToxicityError as e:
            self._log_message_for_verbose(
                f"Found Toxic content..stopping..\n{str(e)}\n"
            )
            raise e
        except ModerationIntentionError as e:
            self._log_message_for_verbose(
                f"Found Harmful intention..stopping..\n{str(e)}\n"
            )
            raise e
        except Exception as e:
            raise e
