from enum import Enum
from typing import List, Type, Any


def get_enum_names(enum_cls: Type[Enum]) -> List[str]:
    """
    Returns a list of names for the given Enum class.

    Args:
        enum_cls: Enum class.

    Returns:
        List of enum names as strings.
    """
    return [member.name for member in enum_cls]


def get_enum_values(enum_cls: Type[Enum]) -> List[Any]:
    """
    Returns a list of values for the given Enum class.

    Args:
        enum_cls: Enum class.

    Returns:
        List of enum values.
    """
    return [member.value for member in enum_cls]
