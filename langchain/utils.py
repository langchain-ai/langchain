"""Generic utility functions."""
import os
from typing import Any, Dict, Optional, Callable
from functools import wraps
import logging

logger = logging.getLogger(__name__)


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

def retry_helper(attempts:Optional[int]=3, sleep:Optional[bool]=True, sleep_time:Optional[int]=2) -> Callable: 
    """Simple retry decorator to avoid boilerplate

    :param attempts: Number of attempts to try the request, defaults to 3
    :type attempts: int, optional
    :param sleep: Whether or not to wait between attempts, defaults to True
    :type sleep: bool, optional
    :param sleep_time: How long to wait between attempts, defaults to 2
    :type sleep_time: int, optional
    :return: Result of the original function
    :rtype: Callable
    """
    def retry_decorator(f:typing.Callable[[str], None]):
        @wraps(f)
        def retry(*args, **kwargs):
            for attempt in range(1,attempts):
                try: 
                    return f(*args, **kwargs)
                except BaseException as e:
                    logger.debug(traceback.format_exc())
                    logger.info(f'Error: {str(e)} Retry {attempt} of {attempts}')
                if sleep: 
                    time.sleep(sleep_time)
            else: 
                raise RuntimeError(f'Retries exceeded for method {f.__name__}')
        return retry
    return retry_decorator
