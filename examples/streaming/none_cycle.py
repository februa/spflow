"""None入力周期と完成出力周期の違いをFlowで接続する例。"""

from __future__ import annotations

from spflow import Flow


class PeriodicMean:
    """入力の有無にかかわらず周期更新し、2周期ごとに平均値を公開する。

    `process`は各周期の`float | None`を受け取り、有効値だけを蓄積する。
    2周期が完成した時点で、その区間に有効値があれば平均値を返す。

    後段処理や周期の駆動は責務に含めず、現在周期の状態更新と完成値の公開だけを担う。
    """

    def __init__(self) -> None:
        self.processed_cycle_count = 0
        self._pending_values: list[float] = []

    def process(self, value: float | None) -> float | None:
        """1周期を処理し、2周期区間が完成した場合だけ平均値を返す。

        Args:
            value: 現在周期の入力値。値が到着しなかった周期は`None`。

        Returns:
            2周期区間に含まれる有効入力の算術平均。区間未完成、または完成区間に
            有効値がない場合は`None`を返し、Flowの後段を呼ばない。

        境界条件:
            `None`入力でも周期数は進める。これにより、入力の到着頻度と状態更新周期を
            分離し、値がない周期を暗黙に飛ばさない。
        """
        self.processed_cycle_count += 1
        if value is not None:
            self._pending_values.append(value)

        if self.processed_cycle_count % 2 != 0:
            return None
        if not self._pending_values:
            # 完成区間に値がなければ、平均値を捏造せず「完成出力なし」として後段を止める。
            return None

        completed_mean = sum(self._pending_values) / len(self._pending_values)
        self._pending_values.clear()
        return completed_mean


def format_completed_mean(value: float) -> str:
    """完成平均値を表示用文字列へ変換する。

    Args:
        value: `PeriodicMean`が公開した完成区間の平均値。

    Returns:
        小数第1位までを含む表示文字列。
    """
    return f"mean={value:.1f}"


def main() -> None:
    """4周期を処理し、現在段と後段で異なる実行回数になることを確認する。"""
    periodic_mean = PeriodicMean()
    cycle_inputs: list[float | None] = [2.0, None, 4.0, 6.0]
    completed_outputs: list[str] = []

    for cycle_input in cycle_inputs:
        # from_value(None)も1周期としてPeriodicMeanへ渡る。
        # PeriodicMeanのNone出力はFlowが0項目へ変換するため、format段は完成時だけ実行される。
        completed_outputs.extend(
            Flow.from_value(cycle_input)
            .map(periodic_mean.process)
            .map(format_completed_mean)
            .to_list()
        )

    if periodic_mean.processed_cycle_count != 4:
        raise RuntimeError("None入力周期が現在段へ通知されていません。")
    if completed_outputs != ["mean=2.0", "mean=5.0"]:
        raise RuntimeError("完成した2周期区間だけが後段へ渡されていません。")

    print(f"processed_cycles={periodic_mean.processed_cycle_count}")
    print(f"completed_outputs={completed_outputs}")


if __name__ == "__main__":
    main()
