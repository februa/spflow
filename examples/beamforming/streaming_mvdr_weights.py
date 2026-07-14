"""FFTまでFlowで運び、MVDR係数設計と信号適用の依存をPythonで明示する例。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from spflow import Flow, FrameBuffer, StepScheduler
from spflow.beamforming import (
    CovarianceEstimator,
    MVDRWeightCallback,
    MVDRWeightSnapshot,
    apply_beamformer_bands,
    design_cbf_coefficients,
)


@dataclass
class AdaptiveBeamformerFlowEnvironment:
    """適応係数設計経路と信号適用経路が共有する状態を保持する。

    入力はFFT後のchannel信号と共分散snapshot、出力は完成係数を適用したbeam信号である。
    frame化、共分散推定、時間分割設計、現在係数の状態だけを保持し、各数式の実装や
    Flowの実行順序を決めることは責務に含めない。
    """

    fft_length: int
    frame_buffer: FrameBuffer
    analysis_window: NDArray[np.float32]
    covariance_estimator: CovarianceEstimator
    steering: NDArray[np.complex64]
    weight_scheduler: StepScheduler[
        MVDRWeightSnapshot,
        int,
        NDArray[np.complex64],
    ]
    active_beamformer_coefficients: NDArray[np.complex64]
    designing_snapshot: MVDRWeightSnapshot | None = None
    waiting_snapshot: MVDRWeightSnapshot | None = None
    next_covariance_generation: int = 0
    active_coefficient_generation: int | None = None
    completed_coefficient_generation: int | None = None
    coefficient_replacement_count: int = 0


def make_environment(
    *, fft_length: int, items_per_cycle: int | None
) -> AdaptiveBeamformerFlowEnvironment:
    """2channel・1beamの適応Flow例を構成する。

    Args:
        fft_length: frame長とFFT長。単位はsample、2以上とする。
        items_per_cycle: 1周期に設計する周波数bin数。`None`は全bin。

    Returns:
        初期固定CBF係数を持つ適応ビームフォーマ環境。

    Raises:
        ValueError: fft_lengthが2未満の場合。
    """
    if fft_length < 2:
        raise ValueError("fft_length must be at least 2.")

    # steering shape: [n_ch=2, n_beam=1, n_band=fft_length]。
    # 全帯域を同相到来とし、h^T a=1の無歪条件を決定論的に検証する。
    steering = np.ones((2, 1, fft_length), dtype=np.complex64)
    return AdaptiveBeamformerFlowEnvironment(
        fft_length=fft_length,
        frame_buffer=FrameBuffer(frame_size=fft_length, hop_size=fft_length, axis=1),
        analysis_window=np.ones(fft_length, dtype=np.float32),
        covariance_estimator=CovarianceEstimator(forgetting_factor=1.0),
        steering=steering,
        weight_scheduler=StepScheduler(
            MVDRWeightCallback(diag_load=1.0e-3),
            items_per_cycle=items_per_cycle,
        ),
        active_beamformer_coefficients=np.asarray(
            design_cbf_coefficients(steering), dtype=np.complex64
        ),
    )


def apply_analysis_window(
    frame: NDArray[Any], window: NDArray[np.float32]
) -> NDArray[np.complex64]:
    """shape`[n_ch,n_sample]`の時間軸へ共通窓を掛ける。"""
    values = np.asarray(frame, dtype=np.complex64)
    if values.ndim != 2 or window.shape != (values.shape[1],):
        raise ValueError("window must agree with frame time axis.")
    # window[None, :]によりchannel軸を保持し、時間sample軸だけをbroadcastする。
    return np.asarray(values * window[np.newaxis, :], dtype=np.complex64)


def calculate_frame_fft(frame: NDArray[Any], fft_length: int) -> NDArray[np.complex64]:
    """時間frameをchannelごとに正規化FFTする。

    Args:
        frame: shape`[n_ch,n_sample]`。axis=1が時間sample。
        fft_length: FFT長。単位はsampleで、n_sample以上とする。

    Returns:
        shape`[n_ch,fft_length]`の複素FFT係数。axis=1が周波数bin。

    Raises:
        ValueError: 入力shapeが不正、またはfft_lengthがframe長未満の場合。
    """
    values = np.asarray(frame, dtype=np.complex64)
    if values.ndim != 2 or fft_length < values.shape[1]:
        raise ValueError("fft_length must cover the 2-D frame time axis.")
    # 共分散経路と信号適用経路で同じ1/N正規化FFT係数を共有する。
    return np.asarray(
        np.fft.fft(values, n=fft_length, axis=1) / fft_length,
        dtype=np.complex64,
    )


def calculate_covariance_snapshot(
    frame_fft: NDArray[Any], env: AdaptiveBeamformerFlowEnvironment
) -> MVDRWeightSnapshot:
    """FFT信号から帯域別共分散とgenerationを持つsnapshotを作る。"""
    spectra = np.asarray(frame_fft, dtype=np.complex64)
    if spectra.shape != (env.steering.shape[0], env.steering.shape[2]):
        raise ValueError("frame_fft and steering must agree on n_ch and n_band.")
    # spectra.T shape: [n_band,n_ch]。各binでR[f]=E[x[f]x[f]^H]を更新する。
    covariance = env.covariance_estimator.process_snapshots(spectra.T)
    generation = env.next_covariance_generation
    env.next_covariance_generation += 1
    return MVDRWeightSnapshot(covariance, env.steering, generation)


def hold_covariance_snapshot_for_adaptive_design(
    snapshot: MVDRWeightSnapshot, env: AdaptiveBeamformerFlowEnvironment
) -> MVDRWeightSnapshot:
    """設計中snapshotを固定し、新着共分散は最新1件だけ待機させる。"""
    env.waiting_snapshot = snapshot
    if env.designing_snapshot is None:
        env.designing_snapshot = env.waiting_snapshot
        env.waiting_snapshot = None
    # 設計中generationを完了まで返し続け、異なる共分散の部分結果を混ぜない。
    return env.designing_snapshot


def calculate_adaptive_beamformer_coefficients(
    snapshot: MVDRWeightSnapshot, env: AdaptiveBeamformerFlowEnvironment
) -> NDArray[np.complex64] | None:
    """MVDR設計を1周期進め、全帯域完成時だけ実適用係数を返す。"""
    result = env.weight_scheduler.process_result(snapshot)
    if not result.updated:
        return None
    generation = snapshot.generation
    if not isinstance(generation, int):
        raise TypeError("MVDR example generation must be int.")
    env.designing_snapshot = None
    env.completed_coefficient_generation = generation
    return result.value


def replace_active_beamformer_coefficients(
    coefficients: NDArray[Any], env: AdaptiveBeamformerFlowEnvironment
) -> None:
    """完成した全帯域係数を信号経路の現在係数へ一括置換する。"""
    completed = np.asarray(coefficients, dtype=np.complex64)
    generation = env.completed_coefficient_generation
    if completed.shape != env.steering.shape:
        raise ValueError("coefficients and steering must have the same shape.")
    if generation is None:
        raise RuntimeError("completed coefficient generation is missing.")
    # copy後に参照を一度で差し替え、信号経路へ帯域ごとの中途状態を見せない。
    env.active_beamformer_coefficients = completed.copy()
    env.active_coefficient_generation = generation
    env.completed_coefficient_generation = None
    env.coefficient_replacement_count += 1


def apply_active_beamformer(
    frame_fft: NDArray[Any], env: AdaptiveBeamformerFlowEnvironment
) -> NDArray[np.complex64]:
    """現在係数との`Y[beam,band]=sum_ch H X`を計算する。"""
    return apply_beamformer_bands(frame_fft, env.active_beamformer_coefficients)


def process_cycle(
    signal_chunk: NDArray[Any], env: AdaptiveBeamformerFlowEnvironment
) -> list[NDArray[np.complex64]]:
    """1処理周期で完成FFT frameごとに係数設計後の信号適用を行う。

    Args:
        signal_chunk: shape`[n_ch,n_sample]`の時間領域入力。
        env: frame化、FFT、共分散、係数設計、現在係数を保持する環境。

    Returns:
        0個・1個・複数個のビーム出力。要素shapeは`[n_beam,n_band]`。

    境界条件:
        frame未完成時は空リストを返す。複数frameが完成した場合は時系列順に処理し、
        各frameで係数設計を先に評価してから完成係数を適用する。
    """
    frame_fft_flow = (
        Flow.from_value(signal_chunk)
        .map(env.frame_buffer.process)
        .map(apply_analysis_window, env.analysis_window)
        .map(calculate_frame_fft, env.fft_length)
    )

    beam_outputs: list[NDArray[np.complex64]] = []
    for frame_fft in frame_fft_flow.to_list():
        # 係数設計経路と信号適用経路には、今回の設計完成状態が同じframeへ適用する
        # 係数を決める依存関係がある。この順序を粗いFlow部品へ隠さずPythonで明示する。
        covariance_snapshot = calculate_covariance_snapshot(frame_fft, env)
        design_snapshot = hold_covariance_snapshot_for_adaptive_design(
            covariance_snapshot,
            env,
        )
        completed_coefficients = calculate_adaptive_beamformer_coefficients(
            design_snapshot,
            env,
        )

        # 全帯域が完成した場合だけactive係数を一括置換する。未完成時は固定CBFまたは
        # 前回完成係数を維持し、中途状態を信号経路へ公開しない。
        if completed_coefficients is not None:
            replace_active_beamformer_coefficients(completed_coefficients, env)

        # Y[beam,band]=Σ_ch H_active[ch,beam,band]X[ch,band]。
        # 上の置換が完了したframeから新係数を使い、未完了なら以前の完成係数を使う。
        beam_outputs.append(apply_active_beamformer(frame_fft, env))

    return beam_outputs


def main() -> None:
    """1周期設計と複数周期設計の係数置換時刻を表示する。"""
    chunks = [
        np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32),
    ]
    for items_per_cycle in (None, 1):
        env = make_environment(fft_length=2, items_per_cycle=items_per_cycle)
        for cycle_index, chunk in enumerate(chunks, start=1):
            outputs = process_cycle(chunk, env)
            print(
                f"items_per_cycle={items_per_cycle} cycle={cycle_index} "
                f"outputs={len(outputs)} active_generation={env.active_coefficient_generation}"
            )


if __name__ == "__main__":
    main()
