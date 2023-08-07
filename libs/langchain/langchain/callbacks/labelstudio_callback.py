import os
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import UUID
from enum import Enum

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import (
    AgentAction, AgentFinish, LLMResult,
    BaseMessage, ChatMessage, HumanMessage, SystemMessage, AIMessage
)


class LabelStudioMode(Enum):
    PROMPT = "prompt"
    CHAT = "chat"


def get_default_label_configs(mode: Union[str, LabelStudioMode]) -> str:
    _default_label_configs = {
        LabelStudioMode.PROMPT.value: '''
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
</View>''',
        LabelStudioMode.CHAT.value: '''
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
</View>'''
    }

    if isinstance(mode, str):
        mode = LabelStudioMode(mode)

    return _default_label_configs[mode.value]


class LabelStudioCallbackHandler(BaseCallbackHandler):
    """Callback Handler that streams data to Label Studio project"""

    DEFAULT_PROJECT_NAME = "LangChain-%Y-%m-%d"

    def __init__(
        self,
        api_key: str = None,
        url: Optional[str] = None,
        project_id: Optional[int] = None,
        project_name: Optional[str] = DEFAULT_PROJECT_NAME,
        project_config: Optional[str] = None,
        mode: Union[str, LabelStudioMode] = LabelStudioMode.PROMPT,
    ):
        super().__init__()

        # Import LabelStudio SDK
        try:
            import label_studio_sdk as ls
        except ImportError:
            raise ImportError(
                f"You're using {self.__class__.__name__} in your code, but you don't have the LabelStudio SDK "
                f"Python package installed or upgraded to the latest version. "
                f"Please run `pip install -U label-studio-sdk` before using this callback."
            )

        # Check if Label Studio API key is provided
        if not api_key:
            if os.getenv("LABEL_STUDIO_API_KEY"):
                api_key = os.getenv("LABEL_STUDIO_API_KEY")
            else:
                raise ValueError(
                    f"You're using {self.__class__.__name__} in your code, Label Studio API key is not provided. "
                    f"Please provide Label Studio API key: go to the Label Studio instance, navigate to "
                    f"Account & Settings -> Access Token and copy the key. "
                    f"Use the key as a parameter for the callback: "
                    f"{self.__class__.__name__}(label_studio_api_key='<your_key_here>', ...) or "
                    f"set the environment variable LABEL_STUDIO_API_KEY=<your_key_here>"
                )
        self.api_key = api_key

        if not url:
            if os.getenv("LABEL_STUDIO_URL"):
                url = os.getenv("LABEL_STUDIO_URL")
            else:
                warnings.warn(
                    f"Label Studio URL is not provided, using default URL: {ls.LABEL_STUDIO_DEFAULT_URL}"
                    f"If you want to provide your own URL, use the parameter: "
                    f"{self.__class__.__name__}(label_studio_url='<your_url_here>', ...) "
                    f"or set the environment variable LABEL_STUDIO_URL=<your_url_here>"
                )
                url = ls.LABEL_STUDIO_DEFAULT_URL
        self.url = url

        # Maps run_id to prompts
        self.payload: Dict[str, Dict] = {}

        self.ls_client = ls.Client(url=self.url,api_key=self.api_key)
        self.project_name = project_name
        self.project_config = project_config or get_default_label_configs(mode)
        self.project_id = project_id or os.getenv("LABEL_STUDIO_PROJECT_ID")
        if project_id is not None:
            self.ls_project = self.ls_client.get_project(int(self.project_id))
        else:
            project_title = datetime.today().strftime(self.project_name)
            existing_projects = self.ls_client.get_projects(title=project_title)
            if existing_projects:
                self.ls_project = existing_projects[0]
                self.project_id = self.ls_project.id
                warnings.warn(
                    f'Project ID not provided. Retrieved project "{project_title}" from the Label Studio instance '
                    f'based on canonical name "{self.project_name}" and the current date (ID={self.ls_project.id}).\n'
                    f'If you want to provide your own project ID, use the parameter: '
                    f'{self.__class__.__name__}(project_id=<your_id_here>, ...) '
                    f'or set the environment variable LABEL_STUDIO_PROJECT_ID=<your_id_here>'
                )
            else:
                self.ls_project = self.ls_client.create_project(title=project_title, label_config=self.project_config)
                self.project_id = self.ls_project.id
                warnings.warn(
                    f'Project ID not provided. Created project "{project_title}" from the Label Studio instance '
                    f'based on canonical name "{self.project_name}" and the current date (ID={self.ls_project.id}).\n'
                    f'If you want to provide your own project ID, use the parameter: '
                    f'{self.__class__.__name__}(project_id=<your_id_here>, ...) '
                    f'or set the environment variable LABEL_STUDIO_PROJECT_ID=<your_id_here>'
                )
        self.parsed_label_config = self.ls_project.parsed_label_config

        # Find the first TextArea tag - we will use the "from_name" and "to_name" from it to create predictions
        self.from_name, self.to_name, self.value = None, None, None
        for tag_name, tag_info in self.parsed_label_config.items():
            if tag_info['type'] == 'TextArea':
                self.from_name = tag_name
                self.to_name = tag_info['to_name'][0]
                self.value = tag_info['inputs'][0]['value']
                break
        if not self.from_name:
            raise ValueError(
                f'Label Studio project "{self.project_name}" does not have a TextArea tag. '
                f'Please add a TextArea tag to the project.'
            )

    def add_prompts_generations(self, run_id: str, generations):
        # Create tasks in Label Studio
        tasks = []
        prompts = self.payload[run_id]['prompts']
        model_version = self.payload[run_id]['kwargs'].get('invocation_params', {}).get('model_name')
        for prompt, generation in zip(prompts, generations):
            tasks.append({
                'data': {
                    self.value: prompt,
                    'run_id': run_id,
                },
                'predictions': [{
                    'result': [{
                        'from_name': self.from_name,
                        'to_name': self.to_name,
                        'type': 'textarea',
                        'value': {
                            'text': [g.text for g in generation]
                        }
                    }],
                    'model_version': model_version
                }]
            })
        self.ls_project.import_tasks(tasks)

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        **kwargs: Any,
    ) -> None:
        """Save the prompts in memory when an LLM starts."""
        run_id = str(kwargs["run_id"])
        self.payload[run_id] = {
            'prompts': prompts,
            'kwargs': kwargs
        }
        print('123123123', self.payload)

    def _get_message_role(self, message: BaseMessage) -> str:
        """Get the role of the message."""
        if isinstance(message, ChatMessage):
            return message.role
        else:
            return message.__class__.__name__

    def on_chat_model_start(self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], *, run_id: UUID,
                            parent_run_id: Optional[UUID] = None, tags: Optional[List[str]] = None,
                            metadata: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
        """Save the prompts in memory when an LLM starts."""
        prompts = []
        for message_list in messages:
            dialog = []
            for message in message_list:
                dialog.append({'role': self._get_message_role(message), 'content': message.content})
            prompts.append(dialog)
        self.payload[str(run_id)] = {
            'prompts': prompts,
            'tags': tags,
            'metadata': metadata,
            'run_id': run_id,
            'parent_run_id': parent_run_id,
            'kwargs': kwargs,
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

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Do nothing when LLM outputs an error."""
        pass

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        pass

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        pass

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
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

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Do nothing when tool outputs an error."""
        pass

    def on_text(self, text: str, **kwargs: Any) -> None:
        """Do nothing"""
        pass

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        """Do nothing"""
        pass
