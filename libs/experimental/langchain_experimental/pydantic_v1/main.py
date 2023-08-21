import typing

# Mypy doesn't handle try-except conditional imports well.
# In the lint environment, pydantic is always v2 (since v2 is supported by langchain)
# so let mypy think that we're always using the `pydantic.v1` namespace in pydantic v2.
if typing.TYPE_CHECKING:
    from pydantic.v1.main import *  # noqa: F403
else:
    try:
        from pydantic.v1.main import *  # noqa: F403
    except ImportError:
        from pydantic.main import *  # noqa: F403
