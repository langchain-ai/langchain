from typing import TYPE_CHECKING, Any, Type

from langchain._api import warn_deprecated

if TYPE_CHECKING:
    from langchain_community.llms.titan_takeoff import TitanTakeoff as TitanTakeoffPro


def _get_titan_pro() -> Type[TitanTakeoffPro]:
    from langchain_community.llms.titan_takeoff import TitanTakeoff as TitanTakeoffPro

    warn_deprecated(
        "0.1.0", "Deprecated in favor of langchain_community.llms.TitanTakeoff."
    )

    return TitanTakeoffPro


def __getattr__(name: str) -> Any:
    if name == "TitanTakeoff":
        return _get_titan_pro()


__all__ = ["TitanTakeoffPro"]
