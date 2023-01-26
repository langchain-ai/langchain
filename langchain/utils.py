"""Generic utility functions."""
import os
from typing import Any, Dict, Optional, Callable, Type
from time import sleep
from functools import wraps
import logging

logger = logging.getLogger()


def get_from_dict_or_env(
    data: Dict[str, Any], key: str, env_key: str, default: Optional[str] = None
) -> str:
    """Get a value from a dictionary or an environment variable."""
    if key in data and data[key]:
        return data[key]
    elif env_key in os.environ and os.environ[env_key]:
        return os.environ[env_key]
    elif default is not None:
        return default
    else:
        raise ValueError(
            f"Did not find {key}, please add an environment variable"
            f" `{env_key}` which contains it, or pass"
            f"  `{key}` as a named parameter."
        )



def retry(ExceptionToCheck: Type[Exception] = Exception, tries: int = 5, delay: float = 0.5, backoff: int = 2, ExceptionToRaise: Type[Exception] =AssertionError) -> Callable:
    """Retry calling the decorated function using an exponential backoff.

    http://www.saltycrane.com/blog/2009/11/trying-out-retry-decorator-python/
    original from: http://wiki.python.org/moin/PythonDecoratorLibrary#Retry

    :param ExceptionToCheck: the exception to check. may be a tuple of
        exceptions to check
    :type ExceptionToCheck: Exception or tuple
    :param tries: number of times to try (not retry) before giving up
    :type tries: int
    :param delay: initial delay between retries in seconds
    :type delay: int
    :param backoff: backoff multiplier e.g. value of 2 will double the delay
        each retry
    :type backoff: int
    :param ExceptionToRaise: exceptions that should be raised instead of retried. may be a tuple.
    :type ExceptionToRaise: Exception or tuple
    """
    def deco_retry(f):   # type: ignore

        @wraps(f)
        def f_retry(*args, **kwargs):  # type: ignore
            mtries, mdelay = tries, delay
            while mtries > 1:
                try:
                    return f(*args, **kwargs)
                except ExceptionToRaise as e:
                    raise e
                except ExceptionToCheck as e:
                    logger.warning(f"Exception: {e}")
                    logger.info(f"retrying in {mdelay} seconds")
                    sleep(mdelay)
                    mtries -= 1
                    mdelay *= backoff
            return f(*args, **kwargs)

        return f_retry  # true decorator

    return deco_retry    
