"""spflow.flow を実装するモジュール。"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


class Flow:
    """0 個・1 個・複数個の値を同一インターフェースで扱う軽量コンテナ。

    パイプライン中で「単一値も複数値も同じ map で流したい」用途を想定する。
    逐次計算グラフの制御補助であり、遅延評価や並列実行は責務に含めない。
    """

    def __init__(self, items: list[Any] | None = None) -> None:
        self._items = [] if items is None else list(items)

    @classmethod
    def empty(cls) -> "Flow":
        """空の Flow を返す。"""
        return cls([])

    @classmethod
    def from_value(cls, value: Any) -> "Flow":
        """単一値を 1 要素 Flow として包む。"""
        return cls([value])

    @classmethod
    def many(cls, items: Iterable[Any]) -> "Flow":
        """反復可能オブジェクトから Flow を作る。"""
        return cls(list(items))

    def map(self, func, *args, **kwargs) -> "Flow":
        """各要素へ関数を適用し、結果を平坦化して新しい Flow を返す。

        `None` は破棄し、`Flow` と `list` は 1 段だけ展開する。これは
        パイプラインの節点が「値なし」「複数値」を返してもそのまま連結できるようにするためである。
        """
        outputs: list[Any] = []
        for item in self._items:
            # None は drop、Flow/list は 1 段展開することで、節点ごとの分岐結果を自然に連結する。
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
        """保持要素を通常の list として返す。"""
        return list(self._items)

    def __repr__(self) -> str:
        return f"Flow({self._items!r})"
