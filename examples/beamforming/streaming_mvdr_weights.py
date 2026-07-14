"""FlowとStepSchedulerで帯域別MVDR係数を時間分割更新する例。"""

from __future__ import annotations

import numpy as np

from spflow import Flow, StepScheduler
from spflow.beamforming import (
    MVDRWeightCallback,
    MVDRWeightSnapshot,
    apply_beamformer_bands,
)


def main() -> None:
    """固定CBF fallbackから完成MVDR係数への切替を確認する。

    入力は2 channel、1 beam、2周波数帯域の決定論的な単一source snapshotである。
    1周期に1帯域ずつ係数を設計し、各周期の最新完成係数を信号へ適用する。
    """
    # steering shape: [n_ch=2, n_beam=1, n_band=2]。
    # 各帯域で異なる到来位相を与え、共役規約を含むh^T a=1を確認する。
    steering = np.array(
        [
            [[1.0 + 0.0j, 1.0 + 0.0j]],
            [[1.0 + 0.0j, 0.0 + 1.0j]],
        ],
        dtype=np.complex64,
    )
    covariance = np.stack(
        [
            np.eye(2, dtype=np.complex64),
            np.diag(np.array([2.0, 1.0], dtype=np.complex64)),
        ],
        axis=0,
    )
    snapshot = MVDRWeightSnapshot(
        covariance=covariance,
        steering=steering,
        generation=0,
    )
    scheduler = StepScheduler(MVDRWeightCallback(diag_load=0.0), items_per_cycle=1)

    # source_snapshot shape: [n_ch=2, n_band=2]。
    # 各帯域で単位振幅sourceがsteeringどおり各channelへ到来した観測を表す。
    source_snapshot = steering[:, 0, :]
    for cycle_index in range(2):
        step_result = Flow.from_value(snapshot).map(scheduler.process_result).to_list()[0]
        beam_output = apply_beamformer_bands(source_snapshot, step_result.value)
        updated_values = (
            Flow.from_value(step_result).map(lambda result: result.updated_value()).to_list()
        )

        # 固定CBF fallbackと完成MVDRの双方が無歪条件を満たし、source振幅1を保存する。
        max_amplitude_error = float(np.max(np.abs(beam_output - 1.0)))
        print(
            f"cycle={cycle_index + 1} updated={step_result.updated} "
            f"updates={len(updated_values)} max_amplitude_error={max_amplitude_error:.3e}"
        )


if __name__ == "__main__":
    main()
