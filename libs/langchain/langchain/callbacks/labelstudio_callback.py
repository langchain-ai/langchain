import os
import warnings
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import UUID

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage, ChatMessage
from langchain_core.outputs import Generation, LLMResult

from langchain.callbacks.base import BaseCallbackHandler


class LabelStudioMode(Enum):
    """Label Studio mode enumerator."""

    PROMPT = "prompt"
    CHAT = "chat"


def get_default_label_configs(
    mode: Union[str, LabelStudioMode]
) -> Tuple[str, LabelStudioMode]:
    """Get default Label Studio configs for the given mode.

    Parameters:
        mode: Label Studio mode ("prompt" or "chat")

    Returns: Tuple of Label Studio config and mode
    """
    _default_label_configs = {
        LabelStudioMode.PROMPT.value: """
<View>
<Style>
    .prompt-box {
        background-color: white;
        border-radius: 10px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        padding: 20px;
    }
</Style>
<View className="root">
    <View className="prompt-box">
        <Text name="prompt" value="$prompt"/>
    </View>
    <TextArea name="response" toName="prompt"
              maxSubmissions="1" editable="true"
              required="true"/>
</View>
<Header value="Rate the response:"/>
<Rating name="rating" toName="prompt"/>
</View>""",
        LabelStudioMode.CHAT.value: """
<View>
<View className="root">
     <Paragraphs name="dialogue"
               value="$prompt"
               layout="dialogue"
               textKey="content"
               nameKey="role"
               granularity="sentence"/>
  <Header value="Final response:"/>
    <TextArea name="response" toName="dialogue"
              maxSubmissions="1" editable="true"
              required="true"/>
</View>
<Header value="Rate the response:"/>
<Rating name="rating" toName="dialogue"/>
</View>""",
    }

    if isinstance(mode, str):
        mode = LabelStudioMode(mode)

    return _default_label_configs[mode.value], mode


class LabelStudioCallbackHandler(BaseCallbackHandler):
    """Label Studio callback handler.
    Provides the ability to send predictions to Label Studio
    for human evaluation, feedback and annotation.

    Parameters:
        api_key: Label Studio API key
        url: Label Studio URL
        project_id: Label Studio project ID
        project_name: Label Studio project name
        project_config: Label Studio project config (XML)
        mode: Label Studio mode ("prompt" or "chat")

    Examples:
        >>> from langchain.llms import OpenAI
        >>> from langchain.callbacks import LabelStudioCallbackHandler
        >>> handler = LabelStudioCallbackHandler(
        ...             api_key='<your_key_here>',
        ...             url='http://localhost:8080',
        ...             project_name='LangChain-%Y-%m-%d',
        ...             mode='prompt'
        ... )
        >>> llm = OpenAI(callbacks=[handler])
        >>> llm.predict('Tell me a story about a dog.')
    """

    DEFAULT_PROJECT_NAME: str = "LangChain-%Y-%m-%d"

    def __init__(
        self,
        api_key: Optional[str] = None,
        url: Optional[str] = None,
        project_id: Optional[int] = None,
        project_name: str = DEFAULT_PROJECT_NAME,
        project_config: Optional[str] = None,
        mode: Union[str, LabelStudioMode] = LabelStudioMode.PROMPT,
    ):
        super().__init__()

        # Import LabelStudio SDK
        try:
            import label_studio_sdk as ls
        except ImportError:
            raise ImportError(
                f"You're using {self.__class__.__name__} in your code,"
                f" but you don't have the LabelStudio SDK "
                f"Python package installed or upgraded to the latest version. "
                f"Please run `pip install -U label-studio-sdk`"
                f" before using this callback."
            )

        # Check if Label Studio API key is provided
        if not api_key:
            if os.getenv("LABEL_STUDIO_API_KEY"):
                api_key = str(os.getenv("LABEL_STUDIO_API_KEY"))
            else:
                raise ValueError(
                    f"You're using {self.__class__.__name__} in your code,"
                    f" Label Studio API key is not provided. "
                    f"Please provide Label Studio API key: "
                    f"go to the Label Studio instance, navigate to "
                    f"Account & Settings -> Access Token and copy the key. "
                    f"Use the key as a parameter for the callback: "
                    f"{self.__class__.__name__}"
                    f"(label_studio_api_key='<your_key_here>', ...) or "
                    f"set the environment variable LABEL_STUDIO_API_KEY=<your_key_here>"
                )
        self.api_key = api_key

        if not url:
            if os.getenv("LABEL_STUDIO_URL"):
                url = os.getenv("LABEL_STUDIO_URL")
            else:
                warnings.warn(
                    f"Label Studio URL is not provided, "
                    f"using default URL: {ls.LABEL_STUDIO_DEFAULT_URL}"
                    f"If you want to provide your own URL, use the parameter: "
                    f"{self.__class__.__name__}"
                    f"(label_studio_url='<your_url_here>', ...) "
                    f"or set the environment variable LABEL_STUDIO_URL=<your_url_here>"
                )
                url = ls.LABEL_STUDIO_DEFAULT_URL
        self.url = url

        # Maps run_id to prompts
        self.payload: Dict[str, Dict] = {}

        self.ls_client = ls.Client(url=self.url, api_key=self.api_key)
        self.project_name = project_name
        if project_config:
            self.project_config = project_config
            self.mode = None
        else:
            self.project_config, self.mode = get_default_label_configs(mode)

        self.project_id = project_id or os.getenv("LABEL_STUDIO_PROJECT_ID")
        if self.project_id is not None:
            self.ls_project = self.ls_client.get_project(int(self.project_id))
        else:
            project_title = datetime.today().strftime(self.project_name)
            existing_projects = self.ls_client.get_projects(title=project_title)
            if existing_projects:
                self.ls_project = existing_projects[0]
                self.project_id = self.ls_project.id
            else:
                self.ls_project = self.ls_client.create_project(
                    title=project_title, label_config=self.project_config
                )
                self.project_id = self.ls_project.id
        self.parsed_label_config = self.ls_project.parsed_label_config

        # Find the first TextArea tag
        # "from_name", "to_name", "value" will be used to create predictions
        self.from_name, self.to_name, self.value, self.input_type = (
            None,
            None,
            None,
            None,
        )
        for tag_name, tag_info in self.parsed_label_config.items():
            if tag_info["type"] == "TextArea":
                self.from_name = tag_name
                self.to_name = tag_info["to_name"][0]
                self.value = tag_info["inputs"][0]["value"]
                self.input_type = tag_info["inputs"][0]["type"]
                break
        if not self.from_name:
            error_message = (
                f'Label Studio project "{self.project_name}" '
                f"does not have a TextArea tag. "
                f"Please add a TextArea tag to the project."
            )
            if self.mode == LabelStudioMode.PROMPT:
                error_message += (
                    "\nHINT: go to project Settings -> "
                    "Labeling Interface -> Browse Templates"
                    ' and select "Generative AI -> '
                    'Supervised Language Model Fine-tuning" template.'
                )
            else:
                error_message += (
                    "\nHINT: go to project Settings -> "
                    "Labeling Interface -> Browse Templates"
                    " and check available templates under "
                    '"Generative AI" section.'
                )
            raise ValueError(error_message)

    def add_prompts_generations(
        self, run_id: str, generations: List[List[Generation]]
    ) -> None:
        # Create tasks in Label Studio
        tasks = []
        prompts = self.payload[run_id]["prompts"]
        model_version = (
            self.payload[run_id]["kwargs"]
            .get("invocation_params", {})
            .get("model_name")
        )
        for prompt, generation in zip(prompts, generations):
            tasks.append(
                {
                    "data": {
                        self.value: prompt,
                        "run_id": run_id,
                    },
                    "predictions": [
                        {
                            "result": [
                                {
                                    "from_name": self.from_name,
                                    "to_name": self.to_name,
                                    "type": "textarea",
                                    "value": {"text": [g.text for g in generation]},
                                }
                            ],
                            "model_version": model_version,
                        }
                    ],
                }
            )
        self.ls_project.import_tasks(tasks)

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        **kwargs: Any,
    ) -> None:
        """Save the prompts in memory when an LLM starts."""
        if self.input_type != "Text":
            raise ValueError(
                f'\nLabel Studio project "{self.project_name}" '
                f"has an input type <{self.input_type}>. "
                f'To make it work with the mode="chat", '
                f"the input type should be <Text>.\n"
                f"Read more here https://labelstud.io/tags/text"
            )
        run_id = str(kwargs["run_id"])
        self.payload[run_id] = {"prompts": prompts, "kwargs": kwargs}

    def _get_message_role(self, message: BaseMessage) -> str:
        """Get the role of the message."""
        if isinstance(message, ChatMessage):
            return message.role
        else:
            return message.__class__.__name__

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Save the prompts in memory when an LLM starts."""
        if self.input_type != "Paragraphs":
            raise ValueError(
                f'\nLabel Studio project "{self.project_name}" '
                f"has an input type <{self.input_type}>. "
                f'To make it work with the mode="chat", '
                f"the input type should be <Paragraphs>.\n"
                f"Read more here https://labelstud.io/tags/paragraphs"
            )

        prompts = []
        for message_list in messages:
            dialog = []
            for message in message_list:
                dialog.append(
                    {
                        "role": self._get_message_role(message),
                        "content": message.content,
                    }
                )
            prompts.append(dialog)
        self.payload[str(run_id)] = {
            "prompts": prompts,
            "tags": tags,
            "metadata": metadata,
            "run_id": run_id,
            "parent_run_id": parent_run_id,
            "kwargs": kwargs,
        }

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Do nothing when a new token is generated."""
        pass

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Create a new Label Studio task for each prompt and generation."""
        run_id = str(kwargs["run_id"])

        # Submit results to Label Studio
        self.add_prompts_generations(run_id, response.generations)

        # Pop current run from `self.runs`
        self.payload.pop(run_id)

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        """Do nothing when LLM outputs an error."""
        pass

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        pass

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        pass

    def on_chain_error(self, error: BaseException, **kwargs: Any) -> None:
        """Do nothing when LLM chain outputs an error."""
        pass

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        """Do nothing when tool starts."""
        pass

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Do nothing when agent takes a specific action."""
        pass

    def on_tool_end(
        self,
        output: str,
        observation_prefix: Optional[str] = None,
        llm_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Do nothing when tool ends."""
        pass

    def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
        """Do nothing when tool outputs an error."""
        pass

    def on_text(self, text: str, **kwargs: Any) -> None:
        """Do nothing"""
        pass

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        """Do nothing"""
        pass
