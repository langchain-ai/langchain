"""
TODO: move this to the base class
"""

from sqlalchemy.exc import InvalidRequestError
from langchain.chains.sql_database.base import SQLDatabaseChain
from langchain.chains.llm import LLMChain
import typing

class ExceptionHandler:
    """
    Class for handling exceptions for the SQLDatabaseChain class.
    
    see list of exceptions here: 
    https://docs.sqlalchemy.org/en/20/core/exceptions.html

    key:
    InvalidRequestError
    """

    #TODO: ask if I should make this a static method

    #TODO: ask about max_tries default value (what it is and how it should be requested)
    #Should be 3 for now (Andy said so), make a constant for it in the class definition in base
    #TODO: could have multiple methods with this name, or make more specific later
    def handle(self, exception:Exception, llm_chain:LLMChain, 
                llm_inputs: dict[str,object], max_tries: int = 3) -> str:
                #TODO: ask about type of llm_inputs -- make sure object works
        if(max_tries == 0): #will eventaully make this self.max_tries and remove from args/use a for loop
            #TODO: ask Andy what output is desired here
            raise Exception("Max tries reached")
        if isinstance(exception, InvalidRequestError): #could be the same as many others
            try:
                return llm_chain.predict(**llm_inputs)
            except Exception as e:
                return self.handle(exception=e, llm_chain=llm_chain,
                                    llm_inputs=llm_inputs, max_tries=max_tries-1)
        else:
            raise exception
    
    #TODO: other implementations of handle, including logical errors