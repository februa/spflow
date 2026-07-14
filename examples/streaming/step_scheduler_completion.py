"""StepSchedulerの完成値利用とFlowによる完成通知を分けて扱う例。"""

from __future__ import annotations

from collections.abc import Hashable, Iterable
from dataclasses import dataclass

from spflow import DoubleBufferCallback, Flow, StepResult, StepScheduler


@dataclass(frozen=True)
class SumSnapshot:
    """時間分割して合計する一世代の入力を表す。

    Attributes:
        values: 加算対象。各要素の単位は任意の整数値単位。
        generation: 異なる入力系列を混ぜないための世代番号。
    """

    values: tuple[int, ...]
    generation: int


class IncrementalSumCallback(DoubleBufferCallback[SumSnapshot, int, int]):
    """SumSnapshotを要素単位で加算し、完成した合計だけを公開する。

    入力は固定されたSumSnapshot、出力は全要素を加算済みの整数である。
    呼び出し周期の決定、完成通知の後段伝播、表示処理は責務に含めない。
    """

    def signature(self, inputs: SumSnapshot) -> Hashable:
        """入力snapshotを識別するgenerationを返す。

        Args:
            inputs: 一世代の加算対象。

        Returns:
            入力系列を区別する整数generation。
        """
        return inputs.generation

    def make_initial_output(self, inputs: SumSnapshot) -> int:
        """初回未完成周期に公開する安全側の合計値0を返す。

        Args:
            inputs: 一世代の加算対象。初期値の決定には使用しない。

        Returns:
            まだ完成値がないことを表す加法単位元0。
        """
        return 0

    def make_work_buffer(self, inputs: SumSnapshot) -> int:
        """新しいgenerationの作業用合計値を0で開始する。

        Args:
            inputs: 一世代の加算対象。作業初期値の決定には使用しない。

        Returns:
            未処理状態の作業用合計値0。
        """
        return 0

    def make_items(self, inputs: SumSnapshot) -> Iterable[int]:
        """加算対象のindex列を作る。

        Args:
            inputs: `values`を持つ一世代のsnapshot。

        Returns:
            0以上`len(values)`未満のindex列。空入力では空range。
        """
        return range(len(inputs.values))

    def update_item(self, item: int, inputs: SumSnapshot) -> None:
        """一つの入力値を作業用合計へ加える。

        Args:
            item: `values`のindex。
            inputs: 周期開始時に固定された入力snapshot。

        Raises:
            RuntimeError: StepSchedulerの開始処理を経ず、作業値が存在しない場合。
            IndexError: itemがvaluesの範囲外の場合。
        """
        current_work = self.work
        if current_work is None:
            raise RuntimeError("work value must be initialized before update.")
        self.work = current_work + inputs.values[item]


def select_newly_completed_value(result: StepResult[int]) -> int | None:
    """StepResultから今回完成した値だけを選ぶ。

    Args:
        result: 最新完成値、更新有無、generationを持つ固定型結果。

    Returns:
        今回`updated=True`なら完成値、未完成周期なら`None`。
    """
    return result.updated_value()


def main() -> None:
    """3回に分割した加算で、安全側の値と完成通知の違いを表示する。"""
    scheduler = StepScheduler(
        IncrementalSumCallback(),
        items_per_cycle=1,
    )
    snapshot = SumSnapshot(values=(2, 4, 6), generation=0)

    for call_index in range(1, 4):
        result = scheduler.process_result(snapshot)

        # result.valueは未完成周期にも安全側の前回完成値を持つ。現在の処理へ毎周期使う値は、
        # Flowへ変換せず、この固定型結果から直接読む。
        active_value = result.value

        # 新しい完成値を保存・通知する独立後段だけは、Noneを0出力へ変換するFlowと接続できる。
        completed_updates = (
            Flow.from_value(result)
            .map(select_newly_completed_value)
            .to_list()
        )
        print(
            f"call={call_index} active={active_value} "
            f"completed_updates={completed_updates}"
        )


if __name__ == "__main__":
    main()
