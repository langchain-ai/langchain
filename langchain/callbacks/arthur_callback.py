import numpy as np
from time import time
from typing import Any, Dict, List, Optional, Union
import uuid

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult

PROMPT_TOKENS = "prompt_tokens"
COMPLETION_TOKENS = "completion_tokens"
TOKEN_USAGE = "token_usage"
FINISH_REASON = "finish_reason"
DURATION = "duration"


class ArthurCallbackHandler(BaseCallbackHandler):
    """Callback Handler that logs to Arthur platform.
    
    Arthur helps enterprise teams optimize model operations and performance at scale. 
    The Arthur API tracks model performance, explainability, and fairness across tabular, NLP, and CV models.
    Our API is model- and platform-agnostic, and continuously scales with complex and dynamic enterprise needs.
    To learn more about Arthur, visit our website at https://www.arthur.ai/ or read the Arthur docs at https://docs.arthur.ai/
    """

    def __init__(
        self,
        model_id: str,
        arthur_url: Optional[str] = "https://app.arthur.ai",
        arthur_login: Optional[str] = None,
        arthur_password: Optional[str] = None
    ) -> None:
        """Initialize callback handler."""

        super().__init__()
        from arthurai import ArthurAI
        from arthurai.common.constants import InputType, OutputType, Stage, ValueType
        from arthurai.common.exceptions import ResponseClientError
        from arthurai.util import generate_timestamps
        
        # save the Arthur timestamp function to be used to create valid inference timestamps in on_llm_end()
        self.timestamp_fn = generate_timestamps
                
        # connect to Arthur               
        if arthur_login is None:
            try:
                arthur_api_key = os.environ['ARTHUR_API_KEY']
            except KeyError:
                raise ValueError('No Arthur authentication provided. Either give a login to the ArthurCallbackHandler \
                    or set an ARTHUR_API_KEY as an environment variable.')
            arthur = ArthurAI(url=arthur_url, access_key=arthur_api_key)
        else:
            if arthur_password is None:
                arthur = ArthurAI(url=arthur_url, login=arthur_login)
            else:
                arthur = ArthurAI(url=arthur_url, login=arthur_login, password=arthur_password)
                
        # get model from Arthur by the provided model ID
        try:
            self.arthur_model = arthur.get_model(model_id)
        except ResponseClientError:
            raise ValueError(f"Was unable to retrieve model with id {model_id} from Arthur. \
                Make sure the ID corresponds to a model that is currently registered with your Arthur account.")
            
        # save the attributes of this model to be used when preparing inferences to log to Arthur in on_llm_end()
        self.attr_names = set([a.name for a in self.arthur_model.get_attributes()])
        self.input_attr = [x for x in self.arthur_model.get_attributes()
                           if x.stage==Stage.ModelPipelineInput and x.value_type==ValueType.Unstructured_Text][0].name
        self.output_attr = [x for x in self.arthur_model.get_attributes()
                           if x.stage==Stage.PredictedValue and x.value_type==ValueType.Unstructured_Text][0].name
        self.token_likelihood_attr = None
        if len([x for x in self.arthur_model.get_attributes() if x.value_type==ValueType.TokenLikelihoods]) > 0:
            self.token_likelihood_attr = [x for x in self.arthur_model.get_attributes()
                           if x.value_type==ValueType.TokenLikelihoods][0].name
        
        # prepare callback data defaults to be updated in on_llm_start()
        self.input_texts: List = []
        self.on_llm_start_time: float = -1.0
        

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """On LLM start, save the input prompts"""
        self.input_texts = prompts
        self.on_llm_start_time = time()

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """On LLM end, send data to Arthur."""
        
        # mark the duration time between on_llm_start() and on_llm_end()
        time_from_start_to_end = time() - self.on_llm_start_time
        
        # create inferences to log to Arthur
        inferences = []
        for i, generations in enumerate(response.generations):
            for generation in generations:

                inference = {
                    'partner_inference_id': str(uuid.uuid4()),
                    'inference_timestamp': self.timestamp_fn(2, '1h', 'now', 'h')[1],
                    self.input_attr: self.input_texts[i], 
                    self.output_attr: generation.text,
                }

                if generation.generation_info is not None:
                    
                    # add finish reason to the inference if the ArthurModel was registered to monitor finish_reason
                    finish_reason = generation.generation_info[FINISH_REASON]
                    if FINISH_REASON in self.attr_names:
                        inference[FINISH_REASON] = finish_reason
                    
                    # add token likelihoods data to the inference if the ArthurModel was registered to monitor token likelihoods
                    logprobs_data = generation.generation_info["logprobs"]
                    if logprobs_data is not None and self.token_likelihood_attr is not None:
                        logprobs = logprobs_data["top_logprobs"]
                        likelihoods = [{k: np.exp(v) for k,v in logprobs[i].items()} for i in range(len(logprobs))]
                        inference[self.token_likelihood_attr] = likelihoods
                
                # add token usage counts to the inference if the ArthurModel was registered to monitor token usage
                print('\n\n\n RESPONSE', response, '\n*****\n')
                if isinstance(response.llm_output, dict) and TOKEN_USAGE in response.llm_output:
                    token_usage = response.llm_output[TOKEN_USAGE]
                    if PROMPT_TOKENS in token_usage and PROMPT_TOKENS in self.attr_names:
                        inference[PROMPT_TOKENS] = token_usage[PROMPT_TOKENS]
                    if COMPLETION_TOKENS in token_usage and COMPLETION_TOKENS in self.attr_names:
                        inference[COMPLETION_TOKENS] = token_usage[COMPLETION_TOKENS]
                
                # add inference duration to the inference if the ArthurModel was registered to monitor inference duration
                if DURATION in self.attr_names:
                    inference[DURATION] = time_from_start_to_end
                
                inferences.append(inference)
                
        # send inferences to arthur
        self.arthur_model.send_inferences(inferences)

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """On chain start, do nothing."""
        pass

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """On chain end, do nothing."""
        pass

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Do nothing when LLM outputs an error."""
        pass

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """On new token, pass."""
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
