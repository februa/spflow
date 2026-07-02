"""spflow.flow を実装するモジュール。"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


class Flow:
    """Lightweight container for zero, one, or many values."""

    def __init__(self, items: list[Any] | None = None) -> None:
        self._items = [] if items is None else list(items)

    @classmethod
    def empty(cls) -> "Flow":
        return cls([])

    @classmethod
    def from_value(cls, value: Any) -> "Flow":
        return cls([value])

    @classmethod
    def many(cls, items: Iterable[Any]) -> "Flow":
        return cls(list(items))

    def map(self, func, *args, **kwargs) -> "Flow":
        outputs: list[Any] = []
        for item in self._items:
            result = func(item, *args, **kwargs)
            if result is None:
                continue
            if isinstance(result, Flow):
                outputs.extend(result._items)
                continue
            if isinstance(result, list):
                outputs.extend(result)
                continue
            outputs.append(result)
        return Flow(outputs)

    def to_list(self) -> list[Any]:
        return list(self._items)

    def __repr__(self) -> str:
        return f"Flow({self._items!r})"
