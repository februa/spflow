"""spflow.callback を実装するモジュール。"""

from __future__ import annotations

from typing import Any


class DoubleBufferCallback:
    """完了済み出力だけを publish するダブルバッファ基底クラス。

    逐次更新中の作業領域 `work` と公開済み値 `prev` を分けることで、
    サイクル途中の不完全な結果を外部へ見せない。スケジューラ連携のための基底であり、
    実際の項目更新ロジックは派生クラスが実装する。
    """

    def __init__(self) -> None:
        self.prev: Any = None
        self.work: Any = None

    def on_start(self, inputs: Any) -> list[Any]:
        """新しいサイクル開始時に作業バッファと項目列を準備する。"""
        self._ensure_initialized(inputs)
        self.work = self.make_work_buffer(inputs)
        return list(self.make_items(inputs))

    def on_step(self, item: Any, inputs: Any) -> None:
        """1 項目分の更新処理を実行する。"""
        if self.work is None:
            self._ensure_initialized(inputs)
            self.work = self.make_work_buffer(inputs)
        self.update_item(item, inputs)

    def on_finish(self, inputs: Any, done: bool) -> Any:
        """サイクル終了時に publish 値を返す。

        `done=True` のときだけ `work` を `prev` へ昇格し、途中結果の露出を防ぐ。
        """
        self._ensure_initialized(inputs)
        # 完了前の work を公開すると中途状態が外部へ漏れるため、done 時だけ昇格する。
        if done:
            self.prev = self.publish(self.work)
        return self.publish(self.prev)

    def reset_cycle(self) -> None:
        """進行中サイクルの作業領域だけを破棄する。"""
        self.work = None

    def signature(self, inputs: Any) -> Any:
        """入力サイクル識別子を返す。

        既定実装は常に `None` を返し、全入力を同一系列として扱う。
        """
        return None

    def publish(self, work: Any) -> Any:
        """公開用オブジェクトを返す。

        `copy()` を持つ配列や辞書状オブジェクトは複製して返し、
        呼び出し側が内部作業バッファを破壊しないようにする。
        """
        if hasattr(work, "copy"):
            return work.copy()
        return work

    def make_initial_output(self, inputs: Any) -> Any:
        """初回 publish 用の初期出力を作る。派生クラスで実装する。"""
        raise NotImplementedError

    def make_work_buffer(self, inputs: Any) -> Any:
        """1 サイクル分の作業バッファを作る。派生クラスで実装する。"""
        raise NotImplementedError

    def make_items(self, inputs: Any):
        """処理対象項目列を返す。派生クラスで実装する。"""
        raise NotImplementedError

    def update_item(self, item: Any, inputs: Any) -> None:
        """単一項目の更新を行う。派生クラスで実装する。"""
        raise NotImplementedError

    def _ensure_initialized(self, inputs: Any) -> None:
        if self.prev is None:
            self.prev = self.make_initial_output(inputs)
