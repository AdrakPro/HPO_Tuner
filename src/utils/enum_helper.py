from enum import Enum
from typing import Any, List, Type


def get_enum_names(enum_cls: Type[Enum]) -> List[str]:
    """
    Returns a list of names for the given Enum class.
    """
    return [member.name for member in enum_cls]


def get_enum_values(enum_cls: Type[Enum]) -> List[Any]:
    """
    Returns a list of values for the given Enum class.
    """
    return [member.value for member in enum_cls]
