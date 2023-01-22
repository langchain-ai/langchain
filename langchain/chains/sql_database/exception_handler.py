"""
TODO: Depending on how many exceptions end up being handled, this could be
moved to a method in the SQLDatabaseChain class.
"""

from sqlalchemy.exc import InvalidRequestError

class ExceptionHandler:
    """
    Class for handling exceptions for the SQLDatabaseChain class.
    
    see list of exceptions here: 
    https://docs.sqlalchemy.org/en/20/core/exceptions.html

    key:
    InvalidRequestError
    """

    #TODO: ask about max_tries default value
    def __init__(self, exception: InvalidRequestError, max_tries: int = 3):
        self.exception = exception


    """Potential handle function:
    switch statment for different types of errors
    if an error occurs again, recursively call function with max_tries - 1
    if max_tries == 0, raise exception (or print something)
    would require passing in the chain object to the ExceptionHandler class
    return new value for result in base.py
    """