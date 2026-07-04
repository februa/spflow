"""spflow.scheduler を実装するモジュール。"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


class StepScheduler:
    """項目列を複数サイクルへ分割して処理するスケジューラ。

    重い処理を 1 回で全部流さず、`items_per_cycle` 件ずつ分割実行することで
    レイテンシを制御する。実際の項目更新内容は callback 側が実装する。
    """

    def __init__(self, callback, items_per_cycle: int | None = None) -> None:
        if items_per_cycle is not None and items_per_cycle <= 0:
            raise ValueError("items_per_cycle must be positive or None.")
        self.callback = callback
        self.items_per_cycle = items_per_cycle
        self._items: list[Any] = []
        self._cursor = 0
        self._active = False
        self._signature = None

    def process(self, inputs: Any) -> Any:
        """入力 1 回分を処理し、必要ならサイクルを継続または完了する。

        Args:
            inputs: callback が参照する任意入力。

        Returns:
            callback が publish した現在出力。

        Notes:
            signature が変わった場合は、旧サイクルの作業内容を破棄して新系列へ切り替える。
            これは異なる共分散やステアリングを跨いで部分更新結果が混ざることを防ぐためである。
        """
        signature = self.callback.signature(inputs)
        # 入力系列が切り替わったら旧サイクルの部分更新結果を破棄し、混線を防ぐ。
        if self._active and signature != self._signature:
            self._reset_cycle()

        if not self._active:
            self._items = list(self.callback.on_start(inputs))
            self._cursor = 0
            self._active = True
            self._signature = signature

        limit = len(self._items) if self.items_per_cycle is None else min(
            self._cursor + self.items_per_cycle,
            len(self._items),
        )

        try:
            while self._cursor < limit:
                item = self._items[self._cursor]
                self.callback.on_step(item, inputs)
                self._cursor += 1
        except Exception:
            self._reset_cycle()
            raise

        done = self._cursor >= len(self._items)
        try:
            output = self.callback.on_finish(inputs, done)
        except Exception:
            self._reset_cycle()
            raise

        if done:
            self._reset_cycle()
        return output

    @staticmethod
    def map(items: Iterable[Any], func, inputs: Any, reducer):
        """項目列へ関数を適用し、reducer で集約する簡易ヘルパー。"""
        results = [func(item, inputs) for item in items]
        return reducer(results, inputs)

    def _reset_cycle(self) -> None:
        self._items = []
        self._cursor = 0
        self._active = False
        self._signature = None
        if hasattr(self.callback, "reset_cycle"):
            self.callback.reset_cycle()
