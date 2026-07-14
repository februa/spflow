"""未完成の作業値を公開しないダブルバッファcallbackを提供する。"""

from __future__ import annotations

from collections.abc import Hashable, Iterable
from copy import copy
from typing import Generic, TypeVar

InputT = TypeVar("InputT")
ItemT = TypeVar("ItemT")
OutputT = TypeVar("OutputT")


class DoubleBufferCallback(Generic[InputT, ItemT, OutputT]):
    """進行中の作業値と公開済み完成値を分離するcallback基底クラス。

    `StepScheduler`から入力snapshotとitemを受け取り、`work`へ逐次反映する。全itemが
    完了した場合だけ`work`を`prev`へ昇格し、未完成周期には前回の完成値を返す。
    item固有の信号処理、処理量の決定、値の後段伝播は責務に含めない。

    型引数は順に入力snapshot型、item型、公開値型を表す。作業値と公開値は同じshape、
    axis、単位を持つことを前提とする。
    """

    def __init__(self) -> None:
        """未初期化の公開bufferと作業bufferを用意する。"""
        self.prev: OutputT | None = None
        self.work: OutputT | None = None

    def on_start(self, inputs: InputT) -> list[ItemT]:
        """新しい入力snapshotの作業bufferとitem列を準備する。

        Args:
            inputs: callback固有の入力snapshot。

        Returns:
            今回処理するitem列。空listは更新対象がない完成周期を表す。

        Raises:
            Exception: 派生クラスの初期値、作業buffer、item生成が失敗した場合。
        """
        self._ensure_initialized(inputs)
        self.work = self.make_work_buffer(inputs)
        return list(self.make_items(inputs))

    def on_step(self, item: ItemT, inputs: InputT) -> None:
        """入力snapshotに対する1 item分の更新を作業bufferへ反映する。

        Args:
            item: `make_items()`が生成した単一item。
            inputs: 周期開始時に固定された入力snapshot。

        Raises:
            Exception: 作業buffer生成または派生クラスのitem更新が失敗した場合。
        """
        if self.work is None:
            self._ensure_initialized(inputs)
            self.work = self.make_work_buffer(inputs)
        self.update_item(item, inputs)

    def on_finish(self, inputs: InputT, done: bool) -> OutputT:
        """完成時だけ作業値を昇格し、外部へ公開可能な値を返す。

        Args:
            inputs: 周期開始時に固定された入力snapshot。
            done: 全itemを処理済みなら`True`。

        Returns:
            `done=True`なら今回完成値、未完成なら前回完成値の独立した浅いcopy。

        Raises:
            RuntimeError: schedulerの開始処理を経ず、必要なbufferが存在しない場合。

        境界条件:
            初回未完成周期にも`make_initial_output()`で作った安全側の完成値を返す。
        """
        self._ensure_initialized(inputs)
        previous = self.prev
        if previous is None:
            raise RuntimeError("initial output was not created.")

        # 未完成のworkを公開すると帯域ごとに世代の異なる係数が見えるため、完了時だけ昇格する。
        if done:
            current_work = self.work
            if current_work is None:
                raise RuntimeError("work buffer was not created.")
            self.prev = self.publish(current_work)
            previous = self.prev
        return self.publish(previous)

    def reset_cycle(self) -> None:
        """進行中周期の作業領域だけを破棄し、前回完成値を保持する。"""
        self.work = None

    def signature(self, inputs: InputT) -> Hashable:
        """入力snapshotのgenerationを返す。

        Args:
            inputs: callback固有の入力snapshot。

        Returns:
            入力系列を区別しない既定値`None`。

        境界条件:
            複数周期中に入力が変わり得るcallbackは必ずoverrideし、明示的なgenerationを返す。
        """
        return None

    def publish(self, work: OutputT) -> OutputT:
        """内部bufferと所有権を分けた公開用objectを返す。

        Args:
            work: 完成済みまたは前回公開済みの値。

        Returns:
            `copy.copy()`による浅いcopy。NumPy配列ではdataを複製するため、呼び出し側の
            in-place更新が内部bufferへ逆流しない。
        """
        return copy(work)

    def make_initial_output(self, inputs: InputT) -> OutputT:
        """初回未完成周期に公開する安全側の完成値を作る。"""
        raise NotImplementedError

    def make_work_buffer(self, inputs: InputT) -> OutputT:
        """1入力snapshotの全itemを格納する作業bufferを作る。"""
        raise NotImplementedError

    def make_items(self, inputs: InputT) -> Iterable[ItemT]:
        """入力snapshotから処理対象item列を作る。"""
        raise NotImplementedError

    def update_item(self, item: ItemT, inputs: InputT) -> None:
        """単一itemの計算結果を`work`へ反映する。"""
        raise NotImplementedError

    def _ensure_initialized(self, inputs: InputT) -> None:
        if self.prev is None:
            self.prev = self.make_initial_output(inputs)
