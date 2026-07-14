"""反復処理を複数周期へ分割する軽量スケジューラを提供する。"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar, runtime_checkable

InputT = TypeVar("InputT")
ItemT = TypeVar("ItemT")
OutputT = TypeVar("OutputT")
CallbackInputT = TypeVar("CallbackInputT", contravariant=True)
CallbackItemT = TypeVar("CallbackItemT")
CallbackOutputT = TypeVar("CallbackOutputT", covariant=True)
MapItemT = TypeVar("MapItemT")
MapResultT = TypeVar("MapResultT")
MapOutputT = TypeVar("MapOutputT")


class _SchedulerCallback(Protocol[CallbackInputT, CallbackItemT, CallbackOutputT]):
    """StepSchedulerが要求するcallback境界を型検査へ伝える。"""

    def signature(self, inputs: CallbackInputT) -> Hashable: ...

    def on_start(self, inputs: CallbackInputT) -> Iterable[CallbackItemT]: ...

    def on_step(self, item: CallbackItemT, inputs: CallbackInputT) -> None: ...

    def on_finish(self, inputs: CallbackInputT, done: bool) -> CallbackOutputT: ...


@runtime_checkable
class _ResettableCallback(Protocol):
    """周期破棄時の任意reset処理を表す内部Protocol。"""

    def reset_cycle(self) -> None: ...


@dataclass(frozen=True)
class StepResult(Generic[OutputT]):
    """StepSchedulerの1回の処理結果を表す。

    `value`は常にcallbackが公開した完成値であり、`updated`は今回の呼び出しで
    新しい一連のitem処理が完了したかを表す。`generation`は入力snapshotの識別子である。
    item処理内容や信号への適用は責務に含めない。

    Attributes:
        value: callbackが公開した完成値。shapeと単位はcallback固有の契約に従う。
        updated: 今回の呼び出しで完成値が更新された場合だけ`True`。
        generation: 今回処理した入力snapshotの識別子。
    """

    value: OutputT
    updated: bool
    generation: Hashable

    def updated_value(self) -> OutputT | None:
        """今回更新された値だけを返す。

        Returns:
            `updated=True`なら`value`、未完成周期なら`None`。

        境界条件:
            `Flow.map()`は`None`を出力なしとして扱うため、このメソッドを介すと
            完成更新があった周期だけを後段へ伝播できる。
        """
        return self.value if self.updated else None


@dataclass
class _ActiveCycle(Generic[InputT, ItemT]):
    """一つの入力snapshotに属する進行中周期を保持する。"""

    inputs: InputT
    items: list[ItemT]
    cursor: int
    generation: Hashable


class StepScheduler(Generic[InputT, ItemT, OutputT]):
    """項目列を複数の呼び出し周期へ分割して処理する。

    入力snapshotから得たitem列を`items_per_cycle`件ずつcallbackへ渡し、重い反復計算の
    時間分割だけを担う。値の0個・1個・複数個伝播、処理内容、完成値の作り方はそれぞれ
    `Flow`、callback、`DoubleBufferCallback`側の責務であり、このクラスには含めない。

    同じgenerationの処理中は、開始時に受け取った`inputs`だけを全itemへ渡す。
    呼び出し側は、そのsnapshotが完成または破棄されるまで内部の可変配列を変更してはならない。
    """

    def __init__(
        self,
        callback: _SchedulerCallback[InputT, ItemT, OutputT],
        items_per_cycle: int | None = None,
    ) -> None:
        """スケジューラを構成する。

        Args:
            callback: item列の生成、1 itemの更新、完成値の公開を実装したcallback。
            items_per_cycle: 1回の`process()`で処理するitem数。`None`は全itemを処理する。

        Raises:
            ValueError: `items_per_cycle`が`None`でも正の整数でもない場合。
        """
        if items_per_cycle is not None and items_per_cycle <= 0:
            raise ValueError("items_per_cycle must be positive or None.")
        self.callback = callback
        self.items_per_cycle = items_per_cycle
        self._cycle: _ActiveCycle[InputT, ItemT] | None = None

    def process(self, inputs: InputT) -> OutputT:
        """入力snapshotの処理を進め、利用可能な最新完成値を返す。

        Args:
            inputs: callback固有の入力snapshot。shape、axis、単位はcallback契約に従う。

        Returns:
            callbackが公開した最新完成値。未完成周期には前回完成値を返す。

        Raises:
            Exception: callback内の例外を、進行中周期を破棄した後にそのまま送出する。

        境界条件:
            generationが変わった場合は旧snapshotの部分結果を破棄する。同じgenerationでは
            開始時のinputsを使い続け、後続呼び出しの別objectと部分結果を混ぜない。
        """
        return self.process_result(inputs).value

    def process_result(self, inputs: InputT) -> StepResult[OutputT]:
        """入力snapshotの処理を進め、更新状態を含む固定型の結果を返す。

        Args:
            inputs: callback固有の入力snapshot。shape、axis、単位はcallback契約に従う。

        Returns:
            最新完成値、今回の更新有無、generationを持つ`StepResult`。

        Raises:
            Exception: callback内の例外を、進行中周期を破棄した後にそのまま送出する。

        境界条件:
            item列が空の場合もその呼び出しで完了とみなし、`updated=True`を返す。
        """
        generation = self.callback.signature(inputs)
        cycle = self._cycle

        # generationが変わった部分結果は、異なる共分散snapshotなどを混ぜないため破棄する。
        if cycle is not None and generation != cycle.generation:
            self._reset_cycle()
            cycle = None

        if cycle is None:
            try:
                items = list(self.callback.on_start(inputs))
            except Exception:
                self._reset_cycle()
                raise
            # 同じgenerationの後続呼び出しで別objectが渡されても、全itemには開始時snapshotを使う。
            cycle = _ActiveCycle(inputs=inputs, items=items, cursor=0, generation=generation)
            self._cycle = cycle

        limit = (
            len(cycle.items)
            if self.items_per_cycle is None
            else min(
                cycle.cursor + self.items_per_cycle,
                len(cycle.items),
            )
        )

        try:
            while cycle.cursor < limit:
                item = cycle.items[cycle.cursor]
                self.callback.on_step(item, cycle.inputs)
                cycle.cursor += 1
        except Exception:
            self._reset_cycle()
            raise

        done = cycle.cursor >= len(cycle.items)
        try:
            output = self.callback.on_finish(cycle.inputs, done)
        except Exception:
            self._reset_cycle()
            raise

        result = StepResult(value=output, updated=done, generation=cycle.generation)
        if done:
            self._reset_cycle()
        return result

    @staticmethod
    def map(
        items: Iterable[MapItemT],
        func: Callable[[MapItemT, InputT], MapResultT],
        inputs: InputT,
        reducer: Callable[[list[MapResultT], InputT], MapOutputT],
    ) -> MapOutputT:
        """項目列を1回で処理し、結果を集約する。

        Args:
            items: 処理対象のitem列。
            func: itemと共通inputsから1 item分の結果を作る関数。
            inputs: 全itemとreducerが参照する共通入力。
            reducer: item結果列を最終出力へまとめる関数。

        Returns:
            reducerが生成した値。shape、axis、単位はfuncとreducerの契約に従う。

        境界条件:
            空のitem列では空listをreducerへ渡す。複数周期への分割は行わない。
        """
        results = [func(item, inputs) for item in items]
        return reducer(results, inputs)

    def _reset_cycle(self) -> None:
        self._cycle = None
        if isinstance(self.callback, _ResettableCallback):
            self.callback.reset_cycle()
