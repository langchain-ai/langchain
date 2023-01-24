"""
TODO: Depending on how many exceptions end up being handled, this could be
moved to a method in the SQLDatabaseChain class.
"""

from sqlalchemy.exc import InvalidRequestError
from langchain.chains.sql_database.base import SQLDatabaseChain
from langchain.chains.llm import LLMChain

class ExceptionHandler:
    """
    Class for handling exceptions for the SQLDatabaseChain class.
    
    see list of exceptions here: 
    https://docs.sqlalchemy.org/en/20/core/exceptions.html

    key:
    InvalidRequestError
    """

    #TODO: ask if I shoudl make this a static method

    #TODO: ask about max_tries default value (what it is and how it should be requested)

    #TODO: could have multiple methods with this name, or make more specific later
    def handle(self, exception:Exception, llm_chain:LLMChain, 
                llm_inputs: dict[str,], max_tries: int = 3) -> str:
        if(max_tries == 0):
            #TODO: ask Andy what output is desired here
            raise Exception("Max tries reached")
        if isinstance(self.exception, InvalidRequestError): #could be the same as many others
            try:
                return llm_chain.predict(**llm_inputs)
            except Exception as e:
                return self.handle(exception=e, llm_chain=llm_chain,
                                    llm_inputs=llm_inputs, max_tries=max_tries-1)
    
    #TODO: other implementations of handle, including logical errors