"""運用受波ベクトルとシェーディング係数から時間領域 SLC を実行する。

このスクリプトは、実運用で使用する受波器位置ベクトル `RECEIVER_POSITION_M`
と周波数別シェーディング係数 `SHADING_COEFFICIENT_BY_CHANNEL_AND_BIN` を
スクリプト内変数として受け、固定整相後の beam-domain 波形を合成してから
既存の `BeamDomainSLC` に渡す。

実運用配列を使う場合は、このファイル上部の `RECEIVER_POSITION_M`、
`SHADING_COEFFICIENT_BY_CHANNEL_AND_BIN`、`SHADING_FREQUENCY_HZ` を差し替える。
これらが `None` の場合だけ、既存 artifacts の運用アレイ JSON と shading JSON を
動作確認用の既定値として読み込む。
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from spflow.beamforming.diagnostic_plotting import centers_to_edges, require_matplotlib
from spflow.beamforming.directions import make_directions
from spflow.beamforming.operational_shading import OperationalShadingDefinition
from spflow.beamforming.operational_sparse_array import OperationalSparseArrayDefinition
from spflow.beamforming.slc import BeamDomainSLC, SlcConfig

FloatArray: TypeAlias = NDArray[np.float64]
ComplexArray: TypeAlias = NDArray[np.complex128]
IntArray: TypeAlias = NDArray[np.int64]
BoolArray: TypeAlias = NDArray[np.bool_]


# ---------------------------------------------------------------------------
# 実運用で差し替える入力変数
# ---------------------------------------------------------------------------

# 受波器位置ベクトル。shape は [n_ch, 3]、単位は m。
# axis=0 が物理 CH、axis=1 が x/y/z 成分である。
RECEIVER_POSITION_M: FloatArray | None = None

# 周波数別シェーディング係数。shape は [n_ch, n_bin]。
# axis=0 は RECEIVER_POSITION_M と同じ物理 CH、axis=1 は SHADING_FREQUENCY_HZ の周波数 bin である。
SHADING_COEFFICIENT_BY_CHANNEL_AND_BIN: FloatArray | None = None

# シェーディング係数の周波数軸。shape は [n_bin]、単位は Hz。
SHADING_FREQUENCY_HZ: FloatArray | None = None


# `RECEIVER_POSITION_M` などが未設定の場合に使う動作確認用の既定ファイル。
DEFAULT_ARRAY_DEFINITION_PATH = Path(
    "artifacts/beamforming/operational_sparse_array/operational_sparse_array_fs32768.json"
)
DEFAULT_SHADING_DEFINITION_PATH = Path(
    "artifacts/beamforming/operational_shading/operational_kaiser_bessel_shading_fs32768.json"
)


# ---------------------------------------------------------------------------
# 実行条件
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path("artifacts/beamforming/operational_vector_shading_slc")

SOUND_SPEED_M_S = 1500.0
FS_HZ = 32768.0
N_BEAM_AZ_REAL = 151
N_BEAM_AZ_VIRTUAL = 0
AZ_MIN_DEG = 0.0
AZ_MAX_DEG = 180.0
ARRAY_SIDE = "right side"

DEFAULT_N_SAMPLE = 32768
DEFAULT_TARGET_FREQUENCY_HZ = 10000.0
DEFAULT_TARGET_AZIMUTH_DEG = 90.0
DEFAULT_TARGET_LEVEL_DB20 = 0.0
DEFAULT_INTERFERER_FREQUENCY_HZ = 8192.0
DEFAULT_INTERFERER_AZIMUTH_DEG = 60.0
DEFAULT_INTERFERER_LEVEL_DB20 = -6.0

BTR_BLOCK_SIZE = 128
BTR_COLOR_RANGE_DB = (-12.0, 0.0)
LEVEL_UNIT_LABEL = "dB re input RMS"
METHOD_ORDER = ("fixed_baseline", "A2_safe", "A2_aggressive")
METHOD_LABELS = {
    "fixed_baseline": "fixed",
    "A2_safe": "A2_safe",
    "A2_aggressive": "A2_aggressive",
}
METHOD_COLORS = {
    "fixed_baseline": "black",
    "A2_safe": "tab:blue",
    "A2_aggressive": "tab:orange",
}
SUMMARY_COLUMNS = (
    "scenario",
    "method",
    "mask_type",
    "candidate",
    "status",
    "source_peak_delta_db",
    "source_azimuth_error_deg",
    "non_source_global_peak_delta_db",
    "non_source_p95_level_delta_db",
    "non_source_p99_level_delta_db",
    "non_source_integrated_level_delta_db",
    "source_to_non_source_margin_delta_db",
    "false_peak_count_delta",
    "max_local_worsening_db_gated",
    "fallback_required",
    "fallback_reason",
    "runtime_factor",
)

SLC_CONFIG = SlcConfig(
    guard=10,
    loading=3.0e-2,
    memory_time_sec=3.0,
    heading_scale_deg=5.0,
    min_ref=8,
    sample_per_dof=5.0,
    tap_len=1,
    eta_normal=1.0,
    eta_limited=1.0,
    enable_heading_forgetting=False,
)


@dataclass(frozen=True)
class RuntimeArrayInput:
    """SLC 前段の固定整相に使う運用入力を保持する。

    このクラスは、受波器位置、周波数別シェーディング係数、周波数軸をまとめる。
    入力は `receiver_position_m[ch, xyz]` と `shading_coefficient[ch, bin]` であり、
    出力は固定整相合成関数が参照する検証済み配列である。

    SLC 係数推定、beam 方位設計、source 波形生成は責務に含めない。
    信号処理上は、物理 CH 空間から beam-domain へ移る前の運用アレイ定義に位置づく。
    """

    receiver_position_m: FloatArray
    shading_coefficient_by_channel_and_bin: FloatArray
    shading_frequency_hz: FloatArray


@dataclass(frozen=True)
class SourceSpec:
    """固定整相入力として合成する単一 narrowband source を表す。

    このクラスは、source の周波数、方位、RMS レベル、初期位相を保持する。
    入力は物理量の scalar であり、出力は beam-domain 波形合成に使う条件である。

    アレイ応答の計算や SLC の target 選択は責務に含めない。
    信号処理上は、受波器ごとの到来位相を決める外部音源条件に位置づく。
    """

    label: str
    frequency_hz: float
    azimuth_deg: float
    level_db20: float
    phase_rad: float


@dataclass(frozen=True)
class ShadedBeamSynthesisResult:
    """シェーディング込み固定整相の合成結果を保持する。

    このクラスは、source を足し合わせた固定整相 beam 出力と、
    target source だけの beam-domain 応答を保持する。

    入力は source 条件、受波器位置、シェーディング係数であり、出力は
    `beam_output[n_beam, n_sample]` と `target_response[n_beam]` である。

    SLC 係数推定、SLC safety gate、診断図の生成は責務に含めない。
    信号処理上は、実運用の固定整相前段を周波数応答として再現する部分に位置づく。
    """

    beam_output: ComplexArray
    target_response: ComplexArray
    selected_bin_by_label: dict[str, int]


def _validate_runtime_array_input(
    receiver_position_m: NDArray[Any],
    shading_coefficient_by_channel_and_bin: NDArray[Any],
    shading_frequency_hz: NDArray[Any],
) -> RuntimeArrayInput:
    """運用受波ベクトルとシェーディング係数の shape と単位前提を検証する。

    Args:
        receiver_position_m: 受波器位置。shape は `[n_ch, 3]`、単位は m。
        shading_coefficient_by_channel_and_bin: シェーディング係数。
            shape は `[n_ch, n_bin]`。axis=0 は物理 CH、axis=1 は周波数 bin である。
        shading_frequency_hz: シェーディング周波数軸。shape は `[n_bin]`、単位は Hz。

    Returns:
        検証済み運用入力。配列 dtype は `float64` である。

    Raises:
        ValueError: shape、CH 数、bin 数、有限性、周波数軸が不正な場合。

    境界条件:
        シェーディング係数の和が 0 になる周波数 bin は、固定整相の正規化ができない。
        そのためここでは係数行列全体の有限性と非負性を検証し、bin ごとの係数和は使用時に確認する。
    """
    positions = np.asarray(receiver_position_m, dtype=np.float64)
    shading = np.asarray(shading_coefficient_by_channel_and_bin, dtype=np.float64)
    frequencies = np.asarray(shading_frequency_hz, dtype=np.float64)

    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("receiver_position_m must have shape (n_ch, 3).")
    if shading.ndim != 2:
        raise ValueError("shading_coefficient_by_channel_and_bin must have shape (n_ch, n_bin).")
    if frequencies.ndim != 1:
        raise ValueError("shading_frequency_hz must have shape (n_bin,).")
    if positions.shape[0] != shading.shape[0]:
        raise ValueError("receiver_position_m and shading coefficient must agree on n_ch.")
    if shading.shape[1] != frequencies.size:
        raise ValueError(
            "shading coefficient bin axis and shading_frequency_hz must agree on n_bin."
        )
    if positions.shape[0] == 0 or frequencies.size == 0:
        raise ValueError("receiver_position_m and shading_frequency_hz must not be empty.")
    if not bool(np.all(np.isfinite(positions))):
        raise ValueError("receiver_position_m must contain finite values.")
    if not bool(np.all(np.isfinite(shading))):
        raise ValueError("shading coefficient must contain finite values.")
    if not bool(np.all(shading >= 0.0)):
        raise ValueError("shading coefficient must be non-negative.")
    if not bool(np.all(np.isfinite(frequencies))):
        raise ValueError("shading_frequency_hz must contain finite values.")
    if not bool(np.all(frequencies > 0.0)):
        raise ValueError("shading_frequency_hz must contain positive values.")
    if not bool(np.all(np.diff(frequencies) >= 0.0)):
        raise ValueError("shading_frequency_hz must be sorted in ascending order.")

    return RuntimeArrayInput(
        receiver_position_m=positions,
        shading_coefficient_by_channel_and_bin=shading,
        shading_frequency_hz=frequencies,
    )


def _load_runtime_array_input() -> RuntimeArrayInput:
    """スクリプト変数または既定 artifact から運用入力を読み込む。

    Returns:
        受波器位置 `[n_ch, 3]`、シェーディング `[n_ch, n_bin]`、周波数軸 `[n_bin]`。

    Raises:
        ValueError: 3 つのスクリプト変数の一部だけが設定されている場合。
        FileNotFoundError: 既定 artifact が存在しない場合。

    境界条件:
        実運用ではスクリプト上部の 3 変数をすべて設定する。
        未設定時の artifact 読み込みは、このリポジトリ内で再現実行するための fallback である。
    """
    user_arrays = (
        RECEIVER_POSITION_M,
        SHADING_COEFFICIENT_BY_CHANNEL_AND_BIN,
        SHADING_FREQUENCY_HZ,
    )
    if any(value is not None for value in user_arrays):
        if any(value is None for value in user_arrays):
            raise ValueError(
                "RECEIVER_POSITION_M, SHADING_COEFFICIENT_BY_CHANNEL_AND_BIN, "
                "SHADING_FREQUENCY_HZ must be set together."
            )
        receiver_position_m = RECEIVER_POSITION_M
        shading_coefficient = SHADING_COEFFICIENT_BY_CHANNEL_AND_BIN
        shading_frequency_hz = SHADING_FREQUENCY_HZ
        if receiver_position_m is None or shading_coefficient is None or shading_frequency_hz is None:
            raise RuntimeError("runtime array input validation did not narrow all three arrays.")
        return _validate_runtime_array_input(
            receiver_position_m=receiver_position_m,
            shading_coefficient_by_channel_and_bin=shading_coefficient,
            shading_frequency_hz=shading_frequency_hz,
        )

    array_definition = OperationalSparseArrayDefinition.load_json(DEFAULT_ARRAY_DEFINITION_PATH)
    shading_definition = OperationalShadingDefinition.load_json(DEFAULT_SHADING_DEFINITION_PATH)

    # 保存済み shading は [n_bin, n_ch] であるため、このスクリプトの運用入力契約
    # [n_ch, n_bin] へ転置する。CH 軸は array_definition.positions_m と一致する。
    return _validate_runtime_array_input(
        receiver_position_m=np.asarray(array_definition.positions_m, dtype=np.float64),
        shading_coefficient_by_channel_and_bin=np.asarray(
            shading_definition.shading_coefficients_by_frequency,
            dtype=np.float64,
        ).T,
        shading_frequency_hz=np.asarray(shading_definition.frequency_grid_hz, dtype=np.float64),
    )


def _direction_from_azimuth_deg(azimuth_deg: float) -> FloatArray:
    """水平面方位を方向余弦 `[x, y, z]` へ変換する。

    Args:
        azimuth_deg: 方位角。単位は deg。

    Returns:
        方向余弦。shape は `[3]`。水平面のみを扱うため z 成分は 0 である。
    """
    azimuth_rad = np.deg2rad(float(azimuth_deg))
    return np.asarray([np.cos(azimuth_rad), np.sin(azimuth_rad), 0.0], dtype=np.float64)


def _select_shading_bin(frequency_hz: float, shading_frequency_hz: FloatArray) -> int:
    """入力周波数に最も近いシェーディング bin index を返す。

    Args:
        frequency_hz: 入力 source 周波数。単位は Hz。
        shading_frequency_hz: シェーディング周波数軸。shape は `[n_bin]`、単位は Hz。

    Returns:
        最近傍 bin index。単位は index。

    Raises:
        ValueError: 周波数が正でない、または Nyquist 周波数を超える場合。

    境界条件:
        実運用の shading は離散 bin で設計済みであるため、ここでは補間せず最近傍を選ぶ。
        補間で係数和や active CH が曖昧になるより、使用 bin を summary に残して追跡可能にする。
    """
    if float(frequency_hz) <= 0.0:
        raise ValueError("frequency_hz must be positive.")
    if float(frequency_hz) >= 0.5 * FS_HZ:
        raise ValueError("frequency_hz must be below Nyquist frequency.")

    frequency_axis = np.asarray(shading_frequency_hz, dtype=np.float64)
    return int(np.argmin(np.abs(frequency_axis - float(frequency_hz))))


def _build_beam_directions() -> tuple[FloatArray, FloatArray]:
    """スクリプト設定の実ビーム数・虚ビーム数から待受方向を作る。

    Returns:
        `(beam_direction, axis_azimuth_deg)`。
        `beam_direction` の shape は `[n_beam, 3]`、`axis_azimuth_deg` の shape は `[n_beam]`。

    境界条件:
        仰角は 0 deg の 1 面に固定する。
        SLC の guard は beam index 単位なので、虚ビームを含む総 beam 軸で処理する。
    """
    direction_3d, axis_azimuth_deg, _ = make_directions(
        az_min_deg=AZ_MIN_DEG,
        az_max_deg=AZ_MAX_DEG,
        el_min_deg=0.0,
        el_max_deg=0.0,
        n_beam_az_real=N_BEAM_AZ_REAL,
        n_beam_az_virtual=N_BEAM_AZ_VIRTUAL,
        n_beam_el=1,
        array_side=ARRAY_SIDE,
        el_preset_deg=[0.0],
    )
    return (
        np.asarray(direction_3d.T, dtype=np.float64),
        np.asarray(axis_azimuth_deg, dtype=np.float64),
    )


def _shaded_fixed_beam_response(
    *,
    receiver_position_m: FloatArray,
    beam_direction: FloatArray,
    channel_shading: FloatArray,
    source_frequency_hz: float,
    source_azimuth_deg: float,
    sound_speed_m_s: float,
) -> ComplexArray:
    """source 方位に対するシェーディング込み固定整相応答を返す。

    Args:
        receiver_position_m: 受波器位置。shape は `[n_ch, 3]`、単位は m。
        beam_direction: 待受方向余弦。shape は `[n_beam, 3]`。
        channel_shading: 使用周波数 bin の CH シェーディング。shape は `[n_ch]`。
        source_frequency_hz: source 周波数。単位は Hz。
        source_azimuth_deg: source 方位。単位は deg。
        sound_speed_m_s: 音速。単位は m/s。

    Returns:
        各待受 beam の複素応答。shape は `[n_beam]`。

    Raises:
        ValueError: shape、係数和、物理パラメータが不正な場合。

    境界条件:
        受波位相は `x_ch(t) = s(t - tau_ch)` として
        `exp(-j 2π f tau_ch)` で表す。固定整相では待受方向の到達遅延
        `tau_beam[ch, beam]` を `exp(+j 2π f tau_beam)` で補償し、
        source 方位と beam 方位が一致したとき応答が 1 になるように正規化する。
    """
    positions = np.asarray(receiver_position_m, dtype=np.float64)
    directions = np.asarray(beam_direction, dtype=np.float64)
    weights = np.asarray(channel_shading, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("receiver_position_m must have shape (n_ch, 3).")
    if directions.ndim != 2 or directions.shape[1] != 3:
        raise ValueError("beam_direction must have shape (n_beam, 3).")
    if weights.ndim != 1 or weights.shape[0] != positions.shape[0]:
        raise ValueError("channel_shading must have shape (n_ch,).")
    if float(np.sum(weights)) <= 0.0:
        raise ValueError("channel_shading must contain positive total weight.")
    if float(sound_speed_m_s) <= 0.0:
        raise ValueError("sound_speed_m_s must be positive.")

    source_direction = _direction_from_azimuth_deg(float(source_azimuth_deg))

    # tau_source[ch] は音響中心に対する source 到達時刻差であり、単位は秒。
    # tau_beam[ch, beam] も同じ符号規約で作り、両者の差が整相後の残留位相になる。
    tau_source_sec = -(positions @ source_direction) / float(sound_speed_m_s)
    tau_beam_sec = -(positions @ directions.T) / float(sound_speed_m_s)

    angular_frequency = 2.0 * np.pi * float(source_frequency_hz)
    arrival_phase = np.exp(-1j * angular_frequency * tau_source_sec)
    steering_phase = np.exp(1j * angular_frequency * tau_beam_sec)

    # steering_phase shape: [n_ch, n_beam]、arrival_phase shape: [n_ch]。
    # CH 軸で shading 付き平均を取り、beam ごとの固定整相応答 [n_beam] を得る。
    weighted_arrival = weights * arrival_phase
    response = steering_phase.T @ weighted_arrival / float(np.sum(weights))
    return np.asarray(response, dtype=np.complex128)


def _synthesize_shaded_beam_output(
    *,
    runtime_input: RuntimeArrayInput,
    beam_direction: FloatArray,
    sources: tuple[SourceSpec, ...],
    target_label: str,
    n_sample: int,
    fs_hz: float,
    sound_speed_m_s: float,
) -> ShadedBeamSynthesisResult:
    """source 条件からシェーディング込み beam-domain 波形を合成する。

    Args:
        runtime_input: 受波器位置と shading 係数。
        beam_direction: 待受方向余弦。shape は `[n_beam, 3]`。
        sources: 合成する source 列。
        target_label: SLC で保護する source の label。
        n_sample: 合成サンプル数。単位は sample。
        fs_hz: サンプリング周波数。単位は Hz。
        sound_speed_m_s: 音速。単位は m/s。

    Returns:
        mixed beam 出力、target source 応答、source ごとの選択 shading bin。

    Raises:
        ValueError: source が空、target label が存在しない、サンプル数が不正な場合。

    境界条件:
        source ごとに周波数が異なる場合、各周波数で最近傍の shading bin を選び、
        その bin の CH 係数で beam 応答を作る。これにより、周波数依存の active CH と
        shading を source ごとの物理入力として反映する。
    """
    if len(sources) == 0:
        raise ValueError("sources must not be empty.")
    if int(n_sample) <= 0:
        raise ValueError("n_sample must be positive.")
    if float(fs_hz) <= 0.0:
        raise ValueError("fs_hz must be positive.")

    time_axis_s = np.arange(int(n_sample), dtype=np.float64) / float(fs_hz)
    n_beam = int(np.asarray(beam_direction).shape[0])
    beam_output = np.zeros((n_beam, int(n_sample)), dtype=np.complex128)
    target_response: ComplexArray | None = None
    selected_bin_by_label: dict[str, int] = {}

    for source in sources:
        bin_index = _select_shading_bin(
            frequency_hz=float(source.frequency_hz),
            shading_frequency_hz=runtime_input.shading_frequency_hz,
        )
        selected_bin_by_label[str(source.label)] = int(bin_index)
        channel_shading = np.asarray(
            runtime_input.shading_coefficient_by_channel_and_bin[:, int(bin_index)],
            dtype=np.float64,
        )
        response = _shaded_fixed_beam_response(
            receiver_position_m=runtime_input.receiver_position_m,
            beam_direction=beam_direction,
            channel_shading=channel_shading,
            source_frequency_hz=float(source.frequency_hz),
            source_azimuth_deg=float(source.azimuth_deg),
            sound_speed_m_s=float(sound_speed_m_s),
        )

        amplitude_rms = float(10.0 ** (float(source.level_db20) / 20.0))
        tone = amplitude_rms * np.exp(
            1j
            * (
                2.0 * np.pi * float(source.frequency_hz) * time_axis_s
                + float(source.phase_rad)
            )
        )
        # response[:, None] shape: [n_beam, 1]、tone[None, :] shape: [1, n_sample]。
        # broadcasting により source ごとの固定整相応答を全 sample へ掛ける。
        beam_output += response[:, np.newaxis] * tone[np.newaxis, :]

        if str(source.label) == str(target_label):
            target_response = response

    if target_response is None:
        raise ValueError("target_label must match one source label.")

    return ShadedBeamSynthesisResult(
        beam_output=beam_output,
        target_response=target_response,
        selected_bin_by_label=selected_bin_by_label,
    )


def _rms_level_db20(signal: NDArray[Any]) -> float:
    """信号の RMS 振幅レベルを dB20 で返す。

    Args:
        signal: 実数または複素信号。任意 shape。

    Returns:
        RMS 振幅レベル。単位は dB re input RMS。
    """
    values = np.asarray(signal)
    rms = float(np.sqrt(np.mean(np.abs(values) ** 2)))
    return float(20.0 * np.log10(max(rms, np.finfo(np.float64).tiny)))



def _rms_levels_db20(beam_output: NDArray[Any]) -> FloatArray:
    """beam ごとの RMS レベルを dB20 で返す。

    Args:
        beam_output: beam-domain 波形。shape は `[n_beam, n_sample]`。
            axis=0 が方位 beam、axis=1 が時間 sample である。

    Returns:
        beam ごとの RMS レベル。shape は `[n_beam]`、単位は dB re input RMS。
    """
    signals = np.asarray(beam_output)
    if signals.ndim != 2:
        raise ValueError("beam_output must have shape (n_beam, n_sample).")

    # SLC 出力は複素になり得るため、abs^2 の平均を power として RMS を求める。
    rms = np.sqrt(np.mean(np.abs(signals) ** 2, axis=1))
    return np.asarray(20.0 * np.log10(np.maximum(rms, np.finfo(np.float64).tiny)))


def _tone_projection_levels_db20(
    *,
    beam_output: NDArray[Any],
    fs_hz: float,
    frequencies_hz: FloatArray,
) -> FloatArray:
    """指定周波数群で beam 出力を複素 tone projection し FRAZ 配列を作る。

    Args:
        beam_output: beam-domain 波形。shape は `[n_beam, n_sample]`。
            axis=0 が方位 beam、axis=1 が時間 sample である。
        fs_hz: サンプリング周波数。単位は Hz。
        frequencies_hz: 評価周波数。shape は `[n_frequency]`、単位は Hz。

    Returns:
        周波数別 beam レベル。shape は `[n_beam, n_frequency]`、単位は dB re input RMS。

    境界条件:
        sweep ではないため、広い STFT 面ではなく source 周波数点だけの小型 FRAZ とする。
    """
    signals = np.asarray(beam_output, dtype=np.complex128)
    if signals.ndim != 2:
        raise ValueError("beam_output must have shape (n_beam, n_sample).")

    frequency_axis = np.asarray(frequencies_hz, dtype=np.float64)
    time_axis_s = np.arange(signals.shape[1], dtype=np.float64) / float(fs_hz)
    levels = np.empty((signals.shape[0], frequency_axis.size), dtype=np.float64)
    for frequency_index, frequency_hz in enumerate(frequency_axis.tolist()):
        reference = np.exp(-1j * 2.0 * np.pi * float(frequency_hz) * time_axis_s)
        # projection[beam] は指定周波数成分の複素振幅であり、合成 tone の RMS 振幅と同じ基準で読む。
        projection = np.mean(signals * reference[np.newaxis, :], axis=1)
        levels[:, frequency_index] = 20.0 * np.log10(
            np.maximum(np.abs(projection), np.finfo(np.float64).tiny)
        )
    return levels


def _btr_relative_levels(
    *,
    beam_output: NDArray[Any],
    fs_hz: float,
    block_size: int,
) -> tuple[FloatArray, FloatArray]:
    """BTR 用に frame max 基準の time-azimuth レベルを作る。

    Args:
        beam_output: beam-domain 波形。shape は `[n_beam, n_sample]`。
        fs_hz: サンプリング周波数。単位は Hz。
        block_size: BTR の時間 frame 長。単位は sample。

    Returns:
        `(time_sec, relative_level_db)`。
        `relative_level_db` の shape は `[n_frame, n_beam]`、単位は dB re frame max。

    境界条件:
        BTR は各 frame の最大 beam を 0 dB に正規化する。
        抑圧量の定量比較ではなく、source track の連続性確認用として保存する。
    """
    signals = np.asarray(beam_output, dtype=np.complex128)
    if signals.ndim != 2:
        raise ValueError("beam_output must have shape (n_beam, n_sample).")
    if int(block_size) <= 0:
        raise ValueError("block_size must be positive.")

    frame_levels: list[FloatArray] = []
    frame_times: list[float] = []
    for start in range(0, signals.shape[1], int(block_size)):
        stop = min(start + int(block_size), signals.shape[1])
        block = signals[:, start:stop]
        if block.shape[1] == 0:
            continue
        rms = np.sqrt(np.mean(np.abs(block) ** 2, axis=1))
        levels = 20.0 * np.log10(np.maximum(rms, np.finfo(np.float64).tiny))
        frame_levels.append(np.asarray(levels - float(np.max(levels)), dtype=np.float64))
        frame_times.append(0.5 * (float(start) + float(stop - 1)) / float(fs_hz))
    return np.asarray(frame_times, dtype=np.float64), np.stack(frame_levels, axis=0)


def _build_source_masks(
    *,
    axis_azimuth_deg: FloatArray,
    sources: tuple[SourceSpec, ...],
    guard_beam_count: int,
) -> tuple[BoolArray, BoolArray, list[int]]:
    """source 方位と guard 幅から source / non-source mask を作る。"""
    axis_deg = np.asarray(axis_azimuth_deg, dtype=np.float64)
    source_mask = np.zeros(axis_deg.size, dtype=np.bool_)
    source_beam_indices: list[int] = []
    for source in sources:
        source_index = int(np.argmin(np.abs(axis_deg - float(source.azimuth_deg))))
        source_beam_indices.append(source_index)
        start = max(0, source_index - int(guard_beam_count))
        stop = min(axis_deg.size, source_index + int(guard_beam_count) + 1)
        # SLC の参照 guard と同じ beam 幅を使い、self-nulling リスクのある領域を
        # non-source 評価から外す。
        source_mask[start:stop] = True
    return source_mask, np.logical_not(source_mask), source_beam_indices


def _mask_runs(mask: BoolArray) -> list[tuple[int, int]]:
    """bool mask の連続 run を返す。"""
    normalized = np.asarray(mask, dtype=np.bool_)
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, enabled in enumerate(normalized.tolist()):
        if enabled and start is None:
            start = index
        elif not enabled and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, int(normalized.size)))
    return runs


def _add_mask_spans(axis: Any, azimuth_deg: FloatArray, source_mask: BoolArray) -> None:
    """plot axis へ source mask / non-source sector を描画する。"""
    azimuth_edges = centers_to_edges(np.asarray(azimuth_deg, dtype=np.float64))
    source = np.asarray(source_mask, dtype=np.bool_)
    for run_index, (start, stop) in enumerate(_mask_runs(np.logical_not(source))):
        axis.axvspan(
            float(azimuth_edges[start]),
            float(azimuth_edges[stop]),
            color="0.92",
            alpha=0.45,
            linewidth=0.0,
            label="non-source sector" if run_index == 0 else None,
        )
    for run_index, (start, stop) in enumerate(_mask_runs(source)):
        axis.axvspan(
            float(azimuth_edges[start]),
            float(azimuth_edges[stop]),
            color="tab:green",
            alpha=0.16,
            linewidth=0.0,
            label="source mask" if run_index == 0 else None,
        )

def _local_source_azimuth_error_deg(
    *,
    levels_db: FloatArray,
    axis_azimuth_deg: FloatArray,
    sources: tuple[SourceSpec, ...],
    search_half_width_beam: int,
) -> float:
    """source 近傍 peak 方位の最大誤差を返す。"""
    errors: list[float] = []
    for source in sources:
        nearest = int(np.argmin(np.abs(axis_azimuth_deg - float(source.azimuth_deg))))
        start = max(0, nearest - int(search_half_width_beam))
        stop = min(axis_azimuth_deg.size, nearest + int(search_half_width_beam) + 1)
        peak_index = int(start + np.argmax(levels_db[start:stop]))
        errors.append(abs(float(axis_azimuth_deg[peak_index]) - float(source.azimuth_deg)))
    return float(max(errors)) if len(errors) > 0 else 0.0


def _integrated_level_delta_db(method_levels: FloatArray, fixed_levels: FloatArray) -> float:
    """non-source 領域の平均 power レベル差を dB で返す。"""
    method_power = np.power(10.0, np.asarray(method_levels, dtype=np.float64) / 10.0)
    fixed_power = np.power(10.0, np.asarray(fixed_levels, dtype=np.float64) / 10.0)
    method_average_power = max(float(np.mean(method_power)), np.finfo(np.float64).tiny)
    fixed_average_power = max(float(np.mean(fixed_power)), np.finfo(np.float64).tiny)
    return float(10.0 * np.log10(method_average_power / fixed_average_power))


def _build_summary_rows(
    *,
    scenario_id: str,
    bl_levels: dict[str, FloatArray],
    axis_azimuth_deg: FloatArray,
    source_mask: BoolArray,
    non_source_mask: BoolArray,
    sources: tuple[SourceSpec, ...],
    fallback_required: bool,
    fallback_reason: str,
    runtime_factor: float,
) -> list[dict[str, object]]:
    """scenario_summary.csv の 3 method row を作る。"""
    rows: list[dict[str, object]] = []
    fixed_levels = bl_levels["fixed_baseline"]
    fixed_source_peak = float(np.max(fixed_levels[source_mask]))
    fixed_non_source_peak = float(np.max(fixed_levels[non_source_mask]))
    fixed_margin = fixed_source_peak - fixed_non_source_peak
    fixed_false_peak_count = int(
        np.count_nonzero(fixed_levels[non_source_mask] > fixed_source_peak - 6.0)
    )

    for method_id in METHOD_ORDER:
        levels = bl_levels[method_id]
        delta = levels - fixed_levels
        source_delta = float(np.max(np.abs(delta[source_mask])))
        non_source_delta = delta[non_source_mask]
        source_peak = float(np.max(levels[source_mask]))
        non_source_peak = float(np.max(levels[non_source_mask]))
        margin = source_peak - non_source_peak
        false_peak_count = int(np.count_nonzero(levels[non_source_mask] > source_peak - 6.0))

        if method_id == "fixed_baseline":
            row_fallback_required = False
            row_fallback_reason = "fixed_baseline_is_always_available"
            status = "fallback_available"
            row_runtime_factor = 0.0
        elif method_id == "A2_safe":
            row_fallback_required = bool(fallback_required)
            row_fallback_reason = fallback_reason
            status = "fallback_to_fixed" if row_fallback_required else "effective_ok"
            row_runtime_factor = float(runtime_factor)
        else:
            row_fallback_required = False
            row_fallback_reason = "diagnostic_raw_candidate_not_used_for_acceptance"
            status = "diagnostic_only_raw"
            row_runtime_factor = float(runtime_factor)

        rows.append(
            {
                "scenario": scenario_id,
                "method": method_id,
                "mask_type": "manual_source_guard",
                "candidate": METHOD_LABELS[method_id],
                "status": status,
                "source_peak_delta_db": source_delta,
                "source_azimuth_error_deg": _local_source_azimuth_error_deg(
                    levels_db=levels,
                    axis_azimuth_deg=axis_azimuth_deg,
                    sources=sources,
                    search_half_width_beam=max(1, int(SLC_CONFIG.guard)),
                ),
                "non_source_global_peak_delta_db": non_source_peak - fixed_non_source_peak,
                "non_source_p95_level_delta_db": float(np.percentile(non_source_delta, 95.0)),
                "non_source_p99_level_delta_db": float(np.percentile(non_source_delta, 99.0)),
                "non_source_integrated_level_delta_db": _integrated_level_delta_db(
                    method_levels=levels[non_source_mask],
                    fixed_levels=fixed_levels[non_source_mask],
                ),
                "source_to_non_source_margin_delta_db": margin - fixed_margin,
                "false_peak_count_delta": int(false_peak_count - fixed_false_peak_count),
                "max_local_worsening_db_gated": float(np.max(non_source_delta)),
                "fallback_required": row_fallback_required,
                "fallback_reason": row_fallback_reason,
                "runtime_factor": row_runtime_factor,
            }
        )
    return rows


def _save_figure(fig: Any, path: Path) -> None:
    """figure を PNG 保存して閉じる。"""
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_bl_overlay(
    *,
    output_path: Path,
    scenario_id: str,
    axis_azimuth_deg: FloatArray,
    source_mask: BoolArray,
    bl_levels: dict[str, FloatArray],
) -> None:
    """fixed / A2_safe / A2_aggressive の BL overlay を保存する。"""
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(10.5, 5.0))
    all_levels = np.concatenate([bl_levels[method] for method in METHOD_ORDER])
    _add_mask_spans(axis, axis_azimuth_deg, source_mask)
    for method_id in METHOD_ORDER:
        axis.plot(
            axis_azimuth_deg,
            bl_levels[method_id],
            linewidth=1.6,
            color=METHOD_COLORS[method_id],
            label=METHOD_LABELS[method_id],
        )
    axis.set_ylim(float(np.min(all_levels) - 1.0), float(np.max(all_levels) + 1.0))
    axis.set_xlabel("Azimuth [deg]")
    axis.set_ylabel(f"RMS Level [{LEVEL_UNIT_LABEL}]")
    axis.set_title(f"{scenario_id}: BL overlay")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    fig.tight_layout()
    _save_figure(fig, output_path)


def _plot_bl_delta(
    *,
    output_path: Path,
    scenario_id: str,
    axis_azimuth_deg: FloatArray,
    source_mask: BoolArray,
    bl_levels: dict[str, FloatArray],
) -> None:
    """A2 - fixed の BL delta を保存する。"""
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(10.5, 4.8))
    fixed = bl_levels["fixed_baseline"]
    max_abs = 1.0
    _add_mask_spans(axis, axis_azimuth_deg, source_mask)
    for method_id in ("A2_safe", "A2_aggressive"):
        delta = bl_levels[method_id] - fixed
        max_abs = max(max_abs, float(np.max(np.abs(delta))))
        axis.plot(
            axis_azimuth_deg,
            delta,
            linewidth=1.5,
            color=METHOD_COLORS[method_id],
            label=f"{METHOD_LABELS[method_id]} - fixed",
        )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_ylim(-(max_abs + 0.5), max_abs + 0.5)
    axis.set_xlabel("Azimuth [deg]")
    axis.set_ylabel("BL Delta [dB re fixed BL level]")
    axis.set_title(f"{scenario_id}: BL delta")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    fig.tight_layout()
    _save_figure(fig, output_path)

def _plot_fraz_delta(
    *,
    output_path: Path,
    scenario_id: str,
    axis_azimuth_deg: FloatArray,
    frequency_hz: FloatArray,
    source_mask: BoolArray,
    fraz_levels: dict[str, FloatArray],
) -> None:
    """A2 - fixed の FRAZ delta を保存する。"""
    import matplotlib.pyplot as plt

    fixed = fraz_levels["fixed_baseline"]
    safe_delta = fraz_levels["A2_safe"] - fixed
    aggressive_delta = fraz_levels["A2_aggressive"] - fixed
    max_abs = max(float(np.max(np.abs(safe_delta))), float(np.max(np.abs(aggressive_delta))), 1.0)
    az_edges = centers_to_edges(axis_azimuth_deg)
    freq_edges = centers_to_edges(frequency_hz)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharey=True)
    image = None
    for axis, method_id, delta in (
        (axes[0], "A2_safe", safe_delta),
        (axes[1], "A2_aggressive", aggressive_delta),
    ):
        image = axis.pcolormesh(
            az_edges,
            freq_edges,
            delta.T,
            shading="flat",
            cmap="coolwarm",
            vmin=-max_abs,
            vmax=max_abs,
        )
        _add_mask_spans(axis, axis_azimuth_deg, source_mask)
        axis.set_title(f"{METHOD_LABELS[method_id]} - fixed")
        axis.set_xlabel("Azimuth [deg]")
    axes[0].set_ylabel("Frequency [Hz]")
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), label="FRAZ Delta [dB re fixed]")
    fig.suptitle(f"{scenario_id}: FRAZ delta")
    fig.subplots_adjust(left=0.07, right=0.86, bottom=0.14, top=0.86, wspace=0.22)
    _save_figure(fig, output_path)


def _plot_btr_panel(
    *,
    output_path: Path,
    scenario_id: str,
    axis_azimuth_deg: FloatArray,
    time_sec: FloatArray,
    source_mask: BoolArray,
    btr_levels: dict[str, FloatArray],
) -> None:
    """fixed / A2_safe / A2_aggressive の BTR panel を保存する。"""
    import matplotlib.pyplot as plt

    az_edges = centers_to_edges(axis_azimuth_deg)
    time_edges = centers_to_edges(time_sec)
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.8), sharey=True)
    image = None
    for axis, method_id in zip(axes, METHOD_ORDER, strict=True):
        image = axis.pcolormesh(
            az_edges,
            time_edges,
            btr_levels[method_id],
            shading="flat",
            cmap="viridis",
            vmin=BTR_COLOR_RANGE_DB[0],
            vmax=BTR_COLOR_RANGE_DB[1],
        )
        _add_mask_spans(axis, axis_azimuth_deg, source_mask)
        axis.set_title(METHOD_LABELS[method_id])
        axis.set_xlabel("Azimuth [deg]")
    axes[0].set_ylabel("Time [s]")
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), label="Relative Level [dB re frame max]")
    fig.suptitle(f"{scenario_id}: BTR source-track continuity")
    fig.text(
        0.5,
        0.01,
        "BTR is dB re frame max; use for source track continuity, not suppression amount.",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    fig.subplots_adjust(left=0.07, right=0.86, bottom=0.16, top=0.86, wspace=0.22)
    _save_figure(fig, output_path)


def _write_csv(path: Path, rows: list[dict[str, object]], columns: tuple[str, ...] | None) -> None:
    """辞書 row 列を CSV 保存する。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        columns if columns is not None else tuple(sorted({key for row in rows for key in row}))
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _row_float(row: dict[str, object], key: str) -> float:
    """CSV rowの数値を型検証して返す。"""
    value = row.get(key, 0.0)
    if not isinstance(value, int | float | str):
        raise TypeError(f"{key} must be numeric or a numeric string.")
    return float(value)


def _build_worst_cases(summary_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """worst_cases.csv の row を作る。"""
    worst_rows: list[dict[str, object]] = []
    metric_names = (
        "source_peak_delta_db",
        "source_azimuth_error_deg",
        "non_source_global_peak_delta_db",
        "non_source_p95_level_delta_db",
        "non_source_p99_level_delta_db",
        "non_source_integrated_level_delta_db",
        "source_to_non_source_margin_delta_db",
        "false_peak_count_delta",
        "max_local_worsening_db_gated",
        "runtime_factor",
    )
    for metric_name in metric_names:
        ranked = sorted(
            summary_rows,
            key=lambda row: abs(_row_float(row, metric_name)),
            reverse=True,
        )
        for rank, row in enumerate(ranked[:10], start=1):
            worst_rows.append(
                {
                    "category": "metric_worst_top10",
                    "metric": metric_name,
                    "rank": rank,
                    "scenario": row.get("scenario", ""),
                    "method": row.get("method", ""),
                    "value": row.get(metric_name, ""),
                    "status": row.get("status", ""),
                    "details": "absolute_value_rank",
                }
            )
    for row in summary_rows:
        if bool(row.get("fallback_required", False)):
            worst_rows.append(
                {
                    "category": "fallback_rows",
                    "metric": "fallback_required",
                    "rank": "",
                    "scenario": row.get("scenario", ""),
                    "method": row.get("method", ""),
                    "value": True,
                    "status": row.get("status", ""),
                    "details": row.get("fallback_reason", ""),
                }
            )
        if str(row.get("status", "")) not in {"effective_ok", "fallback_available"}:
            worst_rows.append(
                {
                    "category": "negative_case_rows",
                    "metric": "status",
                    "rank": "",
                    "scenario": row.get("scenario", ""),
                    "method": row.get("method", ""),
                    "value": row.get("status", ""),
                    "status": row.get("status", ""),
                    "details": "raw candidate or fallback row; adoption uses effective only",
                }
            )
    safe = next(row for row in summary_rows if row["method"] == "A2_safe")
    aggressive = next(row for row in summary_rows if row["method"] == "A2_aggressive")
    for metric_name in metric_names:
        diff = abs(_row_float(aggressive, metric_name) - _row_float(safe, metric_name))
        worst_rows.append(
            {
                "category": "a2_safe_aggressive_large_difference",
                "metric": metric_name,
                "rank": "",
                "scenario": safe.get("scenario", ""),
                "method": "A2_safe_vs_A2_aggressive",
                "value": diff,
                "status": (
                    f"safe={safe.get('status', '')}; "
                    f"aggressive={aggressive.get('status', '')}"
                ),
                "details": f"safe={safe[metric_name]}; aggressive={aggressive[metric_name]}",
            }
        )
    return worst_rows

def _save_review_npz(
    *,
    output_path: Path,
    axis_azimuth_deg: FloatArray,
    frequency_hz: FloatArray,
    time_sec: FloatArray,
    bl_levels: dict[str, FloatArray],
    fraz_levels: dict[str, FloatArray],
    btr_levels: dict[str, FloatArray],
    source_mask: BoolArray,
    non_source_mask: BoolArray,
) -> None:
    """BL / FRAZ / BTR の描画前配列を npz 保存する。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        azimuth_deg=axis_azimuth_deg,
        frequency_hz=frequency_hz,
        time_sec=time_sec,
        fixed_level_db=bl_levels["fixed_baseline"],
        a2_safe_level_db=bl_levels["A2_safe"],
        a2_aggressive_level_db=bl_levels["A2_aggressive"],
        fixed_fraz_level_db=fraz_levels["fixed_baseline"],
        a2_safe_fraz_level_db=fraz_levels["A2_safe"],
        a2_aggressive_fraz_level_db=fraz_levels["A2_aggressive"],
        fixed_btr_level_db=btr_levels["fixed_baseline"],
        a2_safe_btr_level_db=btr_levels["A2_safe"],
        a2_aggressive_btr_level_db=btr_levels["A2_aggressive"],
        source_mask=source_mask,
        non_source_mask=non_source_mask,
    )


def _relative(path: Path, root_dir: Path) -> str:
    """review_pack root からの相対 path を返す。"""
    return path.relative_to(root_dir).as_posix()


def _write_review_index(
    *,
    review_pack_dir: Path,
    scenario_id: str,
    sources: tuple[SourceSpec, ...],
    summary_rows: list[dict[str, object]],
) -> None:
    """review_index.md を保存する。"""
    source_lines = [
        f"  - {source.label}: az={source.azimuth_deg:.3f} deg, "
        f"f={source.frequency_hz:.3f} Hz, level={source.level_db20:.3f} dB20"
        for source in sources
    ]
    review_index_path = review_pack_dir / "review_index.md"
    lines = [
        "# Operational Vector Shading SLC Review Pack",
        "",
        "## 読み方",
        "",
        "- 採否は raw ではなく effective のみで行う。",
        "- fixed_baseline は常に fallback として残す。",
        "- A2_aggressive は safety gate 前の raw candidate であり、採否には使わない。",
        "- BTR は dB re frame max なので、抑圧量の定量比較には使わず、"
        "source track の連続性確認用とする。",
        "- 全図で source mask と non-source sector を背景色で表示する。",
        "",
        "## ファイル",
        "",
        (
            f"- scenario summary: "
            f"`{_relative(review_pack_dir / 'scenario_summary.csv', review_pack_dir)}`"
        ),
        f"- worst cases: `{_relative(review_pack_dir / 'worst_cases.csv', review_pack_dir)}`",
        (
            f"- BL overlay: "
            f"`{_relative(review_pack_dir / 'figures' / 'bl_overlay.png', review_pack_dir)}`"
        ),
        f"- BL delta: `{_relative(review_pack_dir / 'figures' / 'bl_delta.png', review_pack_dir)}`",
        (
            f"- FRAZ delta: "
            f"`{_relative(review_pack_dir / 'figures' / 'fraz_delta.png', review_pack_dir)}`"
        ),
        (
            f"- BTR panel: "
            f"`{_relative(review_pack_dir / 'figures' / 'btr_panel.png', review_pack_dir)}`"
        ),
        (
            f"- plot arrays: "
            f"`{_relative(review_pack_dir / 'data' / 'plot_arrays.npz', review_pack_dir)}`"
        ),
        "",
        "## Scenario",
        "",
        f"- scenario: `{scenario_id}`",
        "- 目的: 運用受波ベクトルとシェーディング係数を用いた単発 SLC 動作確認。",
        "- mask 種別: `manual_source_guard`",
        "- source 方位 / 周波数:",
        *source_lines,
        "",
        "| method | candidate | status | non-source p95 delta | source peak delta | fallback |",
        "|---|---|---|---:|---:|---|",
    ]
    for row in summary_rows:
        lines.append(
            (
                "| `{method}` | `{candidate}` | `{status}` | "
                "{p95:.6f} | {source:.6f} | {fallback} |"
            ).format(
                method=row["method"],
                candidate=row["candidate"],
                status=row["status"],
                p95=_row_float(row, "non_source_p95_level_delta_db"),
                source=_row_float(row, "source_peak_delta_db"),
                fallback=row["fallback_required"],
            )
        )
    review_index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_review_pack(
    *,
    output_dir: Path,
    scenario_id: str,
    sources: tuple[SourceSpec, ...],
    axis_azimuth_deg: FloatArray,
    frequency_hz: FloatArray,
    time_sec: FloatArray,
    source_mask: BoolArray,
    non_source_mask: BoolArray,
    bl_levels: dict[str, FloatArray],
    fraz_levels: dict[str, FloatArray],
    btr_levels: dict[str, FloatArray],
    summary_rows: list[dict[str, object]],
    metadata: dict[str, object],
) -> Path:
    """review_pack 一式を保存する。"""
    require_matplotlib()
    review_pack_dir = Path(output_dir) / "review_pack"
    figure_dir = review_pack_dir / "figures"
    data_dir = review_pack_dir / "data"
    figure_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    _plot_bl_overlay(
        output_path=figure_dir / "bl_overlay.png",
        scenario_id=scenario_id,
        axis_azimuth_deg=axis_azimuth_deg,
        source_mask=source_mask,
        bl_levels=bl_levels,
    )
    _plot_bl_delta(
        output_path=figure_dir / "bl_delta.png",
        scenario_id=scenario_id,
        axis_azimuth_deg=axis_azimuth_deg,
        source_mask=source_mask,
        bl_levels=bl_levels,
    )
    _plot_fraz_delta(
        output_path=figure_dir / "fraz_delta.png",
        scenario_id=scenario_id,
        axis_azimuth_deg=axis_azimuth_deg,
        frequency_hz=frequency_hz,
        source_mask=source_mask,
        fraz_levels=fraz_levels,
    )
    _plot_btr_panel(
        output_path=figure_dir / "btr_panel.png",
        scenario_id=scenario_id,
        axis_azimuth_deg=axis_azimuth_deg,
        time_sec=time_sec,
        source_mask=source_mask,
        btr_levels=btr_levels,
    )
    _save_review_npz(
        output_path=data_dir / "plot_arrays.npz",
        axis_azimuth_deg=axis_azimuth_deg,
        frequency_hz=frequency_hz,
        time_sec=time_sec,
        bl_levels=bl_levels,
        fraz_levels=fraz_levels,
        btr_levels=btr_levels,
        source_mask=source_mask,
        non_source_mask=non_source_mask,
    )
    _write_csv(review_pack_dir / "scenario_summary.csv", summary_rows, SUMMARY_COLUMNS)
    _write_csv(review_pack_dir / "worst_cases.csv", _build_worst_cases(summary_rows), None)
    _write_review_index(
        review_pack_dir=review_pack_dir,
        scenario_id=scenario_id,
        sources=sources,
        summary_rows=summary_rows,
    )
    (review_pack_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return review_pack_dir
def _parse_args() -> argparse.Namespace:
    """CLI 引数を解析する。

    Returns:
        argparse の結果。周波数、方位、レベルは target/interferer それぞれ上書きできる。
    """
    parser = argparse.ArgumentParser(
        description="運用受波ベクトル[nCh,3]とシェーディング[nCh,nBin]から時間領域SLCを実行する。",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--n-sample", type=int, default=DEFAULT_N_SAMPLE)
    parser.add_argument("--duration-s", type=float, default=None)

    parser.add_argument(
        "--frequency-hz",
        "--target-frequency-hz",
        dest="target_frequency_hz",
        type=float,
        default=DEFAULT_TARGET_FREQUENCY_HZ,
    )
    parser.add_argument(
        "--azimuth-deg",
        "--target-azimuth-deg",
        dest="target_azimuth_deg",
        type=float,
        default=DEFAULT_TARGET_AZIMUTH_DEG,
    )
    parser.add_argument(
        "--level-db20",
        "--target-level-db20",
        dest="target_level_db20",
        type=float,
        default=DEFAULT_TARGET_LEVEL_DB20,
    )
    parser.add_argument(
        "--interferer-frequency-hz",
        type=float,
        default=DEFAULT_INTERFERER_FREQUENCY_HZ,
    )
    parser.add_argument(
        "--interferer-azimuth-deg",
        type=float,
        default=DEFAULT_INTERFERER_AZIMUTH_DEG,
    )
    parser.add_argument(
        "--interferer-level-db20",
        type=float,
        default=DEFAULT_INTERFERER_LEVEL_DB20,
    )
    parser.add_argument("--disable-interferer", action="store_true")
    return parser.parse_args()


def main() -> None:
    """スクリプト設定と CLI 上書き値から固定整相 + SLC を実行する。

    境界条件:
        SLC の target beam は target 方位に最も近い待受 beam とする。
        desired response blocking には、同じ受波ベクトル・shading から作った target 応答
        `[n_beam, 1]` を渡し、reference 側に残る target sidelobe を共分散推定前に射影除去する。
    """
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_sample = int(args.n_sample)
    if args.duration_s is not None:
        # duration 指定時は fs から sample 数を決める。
        # 実機ログ長と同じ秒数で評価したい場合に、サンプル数の手計算ミスを避ける。
        n_sample = int(round(float(args.duration_s) * float(FS_HZ)))

    runtime_input = _load_runtime_array_input()
    beam_direction, axis_azimuth_deg = _build_beam_directions()
    target_source = SourceSpec(
        label="target",
        frequency_hz=float(args.target_frequency_hz),
        azimuth_deg=float(args.target_azimuth_deg),
        level_db20=float(args.target_level_db20),
        phase_rad=0.0,
    )
    sources: list[SourceSpec] = [target_source]
    if not bool(args.disable_interferer):
        sources.append(
            SourceSpec(
                label="interferer",
                frequency_hz=float(args.interferer_frequency_hz),
                azimuth_deg=float(args.interferer_azimuth_deg),
                level_db20=float(args.interferer_level_db20),
                phase_rad=0.37,
            )
        )

    synthesis = _synthesize_shaded_beam_output(
        runtime_input=runtime_input,
        beam_direction=beam_direction,
        sources=tuple(sources),
        target_label="target",
        n_sample=n_sample,
        fs_hz=float(FS_HZ),
        sound_speed_m_s=float(SOUND_SPEED_M_S),
    )

    target_beam_index = int(np.argmin(np.abs(axis_azimuth_deg - float(target_source.azimuth_deg))))
    desired_response_matrix = synthesis.target_response[:, np.newaxis]
    slc = BeamDomainSLC(
        n_beam=int(synthesis.beam_output.shape[0]),
        fs_hz=float(FS_HZ),
        block_size=int(n_sample),
        config=SLC_CONFIG,
    )
    slc_start_sec = time.perf_counter()
    slc_result = slc.process(
        beam_output=synthesis.beam_output,
        target_beams=np.array([target_beam_index], dtype=np.int64),
        desired_response_matrix=desired_response_matrix,
    )
    slc_elapsed_sec = time.perf_counter() - slc_start_sec
    input_duration_sec = float(n_sample) / float(FS_HZ)
    runtime_factor = float(slc_elapsed_sec / input_duration_sec)

    fixed_output = np.asarray(synthesis.beam_output, dtype=np.complex128)
    safe_output = fixed_output.copy()
    safe_output[target_beam_index, :] = np.asarray(slc_result.Y[0], dtype=np.complex128)

    aggressive_output = fixed_output.copy()
    if slc_result.W is not None:
        # A2_aggressive は safety gate 前の raw candidate として保存する。
        # W がない場合は参照容量不足で raw が定義できないため fixed fallback と同じ出力にする。
        aggressive_output[target_beam_index, :] = (
            fixed_output[target_beam_index, :]
            - float(SLC_CONFIG.eta_normal) * np.asarray(slc_result.C[0], dtype=np.complex128)
        )

    fallback_reasons: list[str] = []
    if slc_result.W is None:
        fallback_reasons.append("reference_capacity_insufficient")
    if slc_result.safety is not None and bool(slc_result.safety.fallback_required):
        fallback_reasons.extend(str(reason) for reason in slc_result.safety.reasons)
    fallback_required = bool(len(fallback_reasons) > 0)
    fallback_reason = ";".join(fallback_reasons)

    method_outputs = {
        "fixed_baseline": fixed_output,
        "A2_safe": safe_output,
        "A2_aggressive": aggressive_output,
    }
    source_tuple = tuple(sources)
    frequency_hz = np.asarray(
        sorted({float(source.frequency_hz) for source in source_tuple}),
        dtype=np.float64,
    )
    source_mask, non_source_mask, source_beam_indices = _build_source_masks(
        axis_azimuth_deg=axis_azimuth_deg,
        sources=source_tuple,
        guard_beam_count=int(SLC_CONFIG.guard),
    )
    bl_levels = {method: _rms_levels_db20(output) for method, output in method_outputs.items()}
    fraz_levels = {
        method: _tone_projection_levels_db20(
            beam_output=output,
            fs_hz=float(FS_HZ),
            frequencies_hz=frequency_hz,
        )
        for method, output in method_outputs.items()
    }
    time_sec = np.empty(0, dtype=np.float64)
    btr_levels: dict[str, FloatArray] = {}
    for method, output in method_outputs.items():
        method_time_sec, method_btr = _btr_relative_levels(
            beam_output=output,
            fs_hz=float(FS_HZ),
            block_size=int(BTR_BLOCK_SIZE),
        )
        if time_sec.size == 0:
            time_sec = method_time_sec
        btr_levels[method] = method_btr

    scenario_id = (
        f"operational_vector_slc_tf{int(round(target_source.frequency_hz))}_"
        f"ta{int(round(target_source.azimuth_deg)):03d}"
    )
    summary_rows = _build_summary_rows(
        scenario_id=scenario_id,
        bl_levels=bl_levels,
        axis_azimuth_deg=axis_azimuth_deg,
        source_mask=source_mask,
        non_source_mask=non_source_mask,
        sources=source_tuple,
        fallback_required=fallback_required,
        fallback_reason=fallback_reason,
        runtime_factor=runtime_factor,
    )
    metadata: dict[str, object] = {
        "array_input": {
            "receiver_position_shape": [
                int(value) for value in runtime_input.receiver_position_m.shape
            ],
            "shading_coefficient_shape": [
                int(value) for value in runtime_input.shading_coefficient_by_channel_and_bin.shape
            ],
            "shading_frequency_shape": [
                int(value) for value in runtime_input.shading_frequency_hz.shape
            ],
        },
        "parameters": {
            "sound_speed_m_s": float(SOUND_SPEED_M_S),
            "fs_hz": float(FS_HZ),
            "n_beam_az_real": int(N_BEAM_AZ_REAL),
            "n_beam_az_virtual": int(N_BEAM_AZ_VIRTUAL),
            "n_beam_total": int(fixed_output.shape[0]),
            "n_sample": int(n_sample),
            "level_reference": LEVEL_UNIT_LABEL,
        },
        "sources": [source.__dict__ for source in source_tuple],
        "source_beam_indices": [int(index) for index in source_beam_indices],
        "selected_bin_by_label": synthesis.selected_bin_by_label,
        "target_beam": {
            "index": int(target_beam_index),
            "azimuth_deg": float(axis_azimuth_deg[target_beam_index]),
        },
        "slc_config": {
            "guard": int(SLC_CONFIG.guard),
            "loading": float(SLC_CONFIG.loading),
            "loading_reference": "mean diagonal reference covariance power",
            "memory_time_sec": float(SLC_CONFIG.memory_time_sec),
            "heading_scale_deg": float(SLC_CONFIG.heading_scale_deg),
            "min_ref": int(SLC_CONFIG.min_ref),
            "sample_per_dof": float(SLC_CONFIG.sample_per_dof),
            "tap_len": int(SLC_CONFIG.tap_len),
            "eta_normal": float(SLC_CONFIG.eta_normal),
            "enable_output_safety_gate": bool(SLC_CONFIG.enable_output_safety_gate),
        },
        "slc_result": {
            "mode": str(slc_result.mode),
            "eta": float(slc_result.eta),
            "reference_beam_count": int(slc_result.reference_beams.size),
            "capacity": slc_result.capacity.as_dict(),
            "condition_number": None
            if slc_result.covariance_condition_number is None
            else float(slc_result.covariance_condition_number),
            "weight_norm": None
            if slc_result.W is None
            else float(np.linalg.norm(np.asarray(slc_result.W))),
            "uses_desired_response_blocking": bool(
                slc_result.reference_blocking_matrix is not None
            ),
            "safety": None if slc_result.safety is None else slc_result.safety.as_dict(),
            "runtime_factor": float(runtime_factor),
        },
    }
    review_pack_dir = _build_review_pack(
        output_dir=output_dir,
        scenario_id=scenario_id,
        sources=source_tuple,
        axis_azimuth_deg=axis_azimuth_deg,
        frequency_hz=frequency_hz,
        time_sec=time_sec,
        source_mask=source_mask,
        non_source_mask=non_source_mask,
        bl_levels=bl_levels,
        fraz_levels=fraz_levels,
        btr_levels=btr_levels,
        summary_rows=summary_rows,
        metadata=metadata,
    )
    print(json.dumps({"review_pack_dir": str(review_pack_dir.resolve())}, indent=2))

if __name__ == "__main__":
    main()
