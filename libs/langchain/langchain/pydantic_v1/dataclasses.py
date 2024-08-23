try:
    from pydantic.v1.dataclasses import *  # noqa: F403
except ImportError:
    from pydantic.dataclasses import *  # type: ignore # noqa: F403
