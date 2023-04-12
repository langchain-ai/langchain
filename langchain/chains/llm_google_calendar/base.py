"""Chain that interprets a prompt and executes python code to do math."""
import datetime
import json
from typing import Dict, List, Any

from pydantic import BaseModel, Extra, root_validator

from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chains.llm_google_calendar.prompt import CREATE_EVENT_PROMPT, CLASSIFICATION_PROMPT
from langchain.llms.base import BaseLLM
from langchain.prompts.base import BasePromptTemplate
from langchain.python import PythonREPL
from langchain.utilities.google_calendar.loader import google_credentials_loader


class LLMGoogleCalendarChain(Chain, BaseModel):
    """Chain that interprets a prompt and executes python code to do math.

    Example:
        .. code-block:: python

            from langchain import LLMMathChain, OpenAI
            llm_math = LLMMathChain(llm=OpenAI())
    """

    llm: BaseLLM
    """LLM wrapper to use."""
    create_event_prompt: BasePromptTemplate = CREATE_EVENT_PROMPT
    """Prompt to use for creating event."""
    classification_prompt: BasePromptTemplate = CLASSIFICATION_PROMPT
    """Prompt to use for classification."""

    query:str
    query_input_key: str = "query"  #: :meta private:
    date_input_key: str = "date"  #: :meta private:
    u_timezone_input_key: str = "u_timezone"  #: :meta private:

    service: Any  #: :meta private:
    google_http_error: Any  #: :meta private:
    creds: Any  #: :meta private:


    output_key: str = "answer"  #: :meta private:

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @classmethod
    def from_default(cls, query: str) -> LLMGoogleCalendarChain:
        """Load with default LLM."""
        return cls(llm=cls.llm, query=query)


    # @property
    # def input_keys(self) -> List[str]:
    #     """Expect input key.

    #     :meta private:
    #     """
    #     return [self.query_input_key, self.date_input_key, self.u_timezone_input_key]

    # @property
    # def output_keys(self) -> List[str]:
    #     """Expect output key.

    #     :meta private:
    #     """
    #     return [self.output_key]

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        #
        # Auth done through OAuth2.0

        try:
            from langchain.utilities.google_calendar.loader import google_credentials_loader
            # save the values from loader to values
            values.update(google_credentials_loader())
            
        except ImportError:
            raise ValueError(
                "Could not import google python packages. "
                """Please it install it with `pip install --upgrade
                google-api-python-client google-auth-httplib2 google-auth-oauthlib`."""
            )
        return values



    def run_classification(self, query: str) -> str:
        """Run classification on query."""
        from langchain import LLMChain, OpenAI, PromptTemplate

        prompt = PromptTemplate(
            template=CLASSIFICATION_PROMPT, input_variables=["query"]
        )
        llm_chain = LLMChain(
            llm=OpenAI(temperature=0, model="text-davinci-003"),
            prompt=prompt,
            verbose=True,
        )
        return llm_chain.run(query=query).strip().lower()

    def run_create_event(self, query: str) -> str:
        create_event_chain = LLMChain(
            llm=self.llm,
            prompt=self.create_event_prompt,
            verbose=True,
        )
        date = datetime.datetime.utcnow().isoformat() + "Z"
        u_timezone = str(
            datetime.datetime.now(datetime.timezone.utc).astimezone().tzinfo
        )
        
        date = datetime.datetime.utcnow().isoformat() + "Z"
        u_timezone = datetime.datetime.now(datetime.timezone.utc).astimezone().tzinfo
        output = create_event_chain.run(
            query=query, date=date, u_timezone=u_timezone
        ).strip()

        loaded = json.loads(output)
        (
            event_summary,
            event_start_time,
            event_end_time,
            event_location,
            event_description,
            user_timezone,
        ) = loaded.values()

        event = self.create_event(
            event_summary=event_summary,
            event_start_time=event_start_time,
            event_end_time=event_end_time,
            user_timezone=user_timezone,
            event_location=event_location,
            event_description=event_description,
        )
        return "Event created successfully, details: event " + event.get("htmlLink")

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        output = ""
        classification = self.run_classification(self.query)
        if classification == "create_event":
            output = self.run_create_event(query=self.query)

        
        return {self.output_key: output}
