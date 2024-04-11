"""Util that calls Dataherald."""
from typing import Any, Dict, Optional

from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.utils import get_from_dict_or_env


class DataheraldAPIWrapper(BaseModel):
    """Wrapper for Dataherald.

    Docs for using:

    1. Go to dataherald and sign up
    2. Create an API key
    3. Save your API key into DATAHERALD_API_KEY env variable
    4. pip install dataherald

    """

    dataherald_client: Any  #: :meta private:
    db_connection_id: str
    dataherald_api_key: Optional[str] = None

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        dataherald_api_key = get_from_dict_or_env(
            values, "dataherald_api_key", "DATAHERALD_API_KEY"
        )
        values["dataherald_api_key"] = dataherald_api_key

        try:
            import dataherald

        except ImportError:
            raise ImportError(
                "dataherald is not installed. "
                "Please install it with `pip install dataherald`"
            )

        client = dataherald.Dataherald(api_key=dataherald_api_key)
        values["dataherald_client"] = client

        return values

    def run(self, prompt: str) -> str:
        """Generate a sql query through Dataherald and parse result."""
        from dataherald.types.sql_generation_create_params import Prompt

        prompt_obj = Prompt(text=prompt, db_connection_id=self.db_connection_id)
        res = self.dataherald_client.sql_generations.create(prompt=prompt_obj)

        try:
            answer = res.sql
            if not answer:
                # We don't want to return the assumption alone if answer is empty
                return "No answer"
            else:
                return f"Answer: {answer}"

        except StopIteration:
            return "Dataherald wasn't able to answer it"
