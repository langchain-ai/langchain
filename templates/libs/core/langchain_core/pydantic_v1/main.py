try:
    from pydantic.v1.main import *  # noqa: F403
except ImportError:
    from pydantic.main import *  # noqa: F403
