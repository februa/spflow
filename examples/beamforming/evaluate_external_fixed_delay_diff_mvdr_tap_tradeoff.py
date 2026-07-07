"""外部アレイ・shading・小数遅延 FIR による差分 FIR tap tradeoff 評価。"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from examples.beamforming.external_fixed_delay_diff_mvdr_inputs import (  # noqa: E402
    apply_frequency_shading_to_weights,
    load_complex_shading_matlab_raw,
    load_fractional_delay_filter_bank_matlab_raw,
    load_fractional_delay_filter_bank_npz,
    load_positions_matlab_raw,
    select_shading_for_frequencies,
)
from spflow.beamforming import (  # noqa: E402
    DelayTable,
    DifferenceCorrectionFIR,
    DifferenceCorrectionFIRDesigner,
    LoadedMVDRWeightDesigner,
    design_fixed_delay_fractional_weights_from_delay_table,
    make_directions,
)
from spflow.beamforming.time_delay import FractionalDelayFilterBank  # noqa: E402

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class ExternalTapTradeoffConfig:
    """外部アレイ係数を使う tap 数 tradeoff 評価条件。

    このクラスは、評価周波数、beam grid、source 条件、処理量測定条件を保持する。
    入力は ndarray 化済み外部係数と組み合わせて使う scalar 設定であり、出力は CSV/Markdown の
    行に集約される。外部 raw file の読み込みや図化は責務に含めない。
    信号処理上は、差分 FIR の周波数応答近似誤差と時間領域 FIR 処理量の評価条件である。
    """

    fs_hz: float = 32768.0
    sound_speed_m_s: float = 1500.0
    source_azimuth_deg: float = 60.0
    source_level_db20: float = 0.0
    noise_power_per_channel: float = 1.0e-2
    frequency_min_hz: float = 768.0
    frequency_max_hz: float = 6144.0
    frequency_step_hz: float = 128.0
    az_min_deg: float = 0.0
    az_max_deg: float = 180.0
    n_beam_az_real: int = 121
    diagonal_loading_ratio: float = 1.0e-2
    benchmark_sample_count: int = 4096
    benchmark_repeats: int = 5
    random_seed: int = 20260707


@dataclass(frozen=True)
class ExternalTapTradeoffRow:
    """tap 数 1 条件の精度・処理量 metric を保持する。"""

    fir_taps: int
    frequency_bin_count: int
    beam_count: int
    channel_count: int
    mac_per_sample_per_beam: int
    mac_factor_re_128: float
    measured_us_per_sample_per_beam: float
    measured_runtime_factor_re_128: float
    max_q_reconstruction_rms_error: float
    max_q_reconstruction_abs_error: float
    max_target_response_abs_error: float
    max_final_weight_rms_error: float


def _direction_from_azimuth(azimuth_deg: float) -> FloatArray:
    """水平面 azimuth [deg] から direction cosine `[3]` を作る。"""
    azimuth_rad = np.deg2rad(float(azimuth_deg))
    return np.array([np.cos(azimuth_rad), np.sin(azimuth_rad), 0.0], dtype=np.float64)


def _arrival_steering(
    array_positions_m: FloatArray,
    azimuth_deg: float,
    frequencies_hz: FloatArray,
    sound_speed_m_s: float,
) -> ComplexArray:
    """source 方位の arrival steering を `[n_freq, n_ch]` で返す。

    Args:
        array_positions_m: センサ位置。shape は `[n_ch, 3]`、単位は m。
        azimuth_deg: source 方位。単位は deg。
        frequencies_hz: 評価周波数。shape は `[n_freq]`、単位は Hz。
        sound_speed_m_s: 音速。単位は m/s。

    Returns:
        arrival steering。shape は `[n_freq, n_ch]`。
    """
    direction = _direction_from_azimuth(float(azimuth_deg))
    # tau[ch] = r_ch dot u / c。基準点に対する到達時刻差 [s] を表す。
    tau_sec = array_positions_m @ direction / float(sound_speed_m_s)
    # exp(-j 2π f tau) は、source 到来波が各 channel に持つ位相差である。
    phase = -1j * 2.0 * np.pi * frequencies_hz[:, np.newaxis] * tau_sec[np.newaxis, :]
    return np.asarray(np.exp(phase), dtype=np.complex128)


def _make_source_covariance(
    source_steering: ComplexArray,
    *,
    source_level_db20: float,
    noise_power_per_channel: float,
) -> ComplexArray:
    """source を含む単一 source + チャネル無相関雑音の共分散を返す。"""
    n_freq, n_ch = source_steering.shape
    covariance = np.zeros((n_freq, n_ch, n_ch), dtype=np.complex128)
    source_power = float(10.0 ** (float(source_level_db20) / 10.0))
    for frequency_index in range(n_freq):
        steering = source_steering[frequency_index]
        # R[k] = sigma_s^2 a_s[k] a_s[k]^H + sigma_n^2 I。
        covariance[frequency_index] = source_power * np.outer(steering, steering.conj())
        covariance[frequency_index] += float(noise_power_per_channel) * np.eye(
            n_ch, dtype=np.complex128
        )
    return covariance


def _measure_fir_runtime(
    *,
    fir_taps: int,
    n_ch: int,
    sample_count: int,
    repeats: int,
    rng: np.random.Generator,
) -> float:
    """`DifferenceCorrectionFIR.process` の 1 sample あたり実測秒数を返す。"""
    input_signal = np.asarray(
        rng.standard_normal((n_ch, sample_count)) + 1j * rng.standard_normal((n_ch, sample_count)),
        dtype=np.complex128,
    )
    taps = np.asarray(
        rng.standard_normal((n_ch, fir_taps)) + 1j * rng.standard_normal((n_ch, fir_taps)),
        dtype=np.complex128,
    )
    fir = DifferenceCorrectionFIR(n_ch=n_ch, fir_taps=fir_taps)
    fir.update_coefficients(taps)
    # 初回は履歴初期化と Python 側の cache 影響を含むため、測定前に捨てる。
    fir.process(input_signal)
    elapsed_values: list[float] = []
    for _ in range(int(repeats)):
        fir.reset()
        start = time.perf_counter()
        fir.process(input_signal)
        elapsed_values.append(time.perf_counter() - start)
    return float(np.median(np.asarray(elapsed_values, dtype=np.float64)) / float(sample_count))


def evaluate_external_fir_tap_tradeoff(
    *,
    array_positions_m: NDArray[Any],
    shading_by_channel_bin: NDArray[Any],
    shading_frequency_step_hz: float,
    fractional_delay_filter_bank: FractionalDelayFilterBank,
    tap_counts: tuple[int, ...] = (16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512),
    config: ExternalTapTradeoffConfig = ExternalTapTradeoffConfig(),
) -> list[ExternalTapTradeoffRow]:
    """外部 ndarray を使い、tap 数ごとの FIR 化誤差と処理量を評価する。

    Args:
        array_positions_m: 実アレイ位置。shape は `[n_ch, 3]`、単位は m。
        shading_by_channel_bin: 複素 channel shading。shape は `[n_ch, n_shading_bin]`。
        shading_frequency_step_hz: shading bin 間隔。単位は Hz。
        fractional_delay_filter_bank: 小数遅延 FIR バンク。
        tap_counts: 評価する差分 FIR tap 数。
        config: 評価条件。

    Returns:
        tap 数ごとの metric 行。
    """
    positions = np.asarray(array_positions_m, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("array_positions_m must have shape [n_ch, 3].")
    frequencies_hz = np.arange(
        float(config.frequency_min_hz),
        float(config.frequency_max_hz) + 0.5 * float(config.frequency_step_hz),
        float(config.frequency_step_hz),
        dtype=np.float64,
    )
    directions, _, _ = make_directions(
        az_min_deg=float(config.az_min_deg),
        az_max_deg=float(config.az_max_deg),
        el_min_deg=0.0,
        el_max_deg=0.0,
        n_beam_az_real=int(config.n_beam_az_real),
        n_beam_az_virtual=0,
        n_beam_el=1,
        array_side="right side",
        el_preset_deg=[0.0],
    )
    beam_directions = directions.T.astype(np.float64)
    delay_table = DelayTable.from_geometry(
        array_pos_m=positions,
        dir_cos=beam_directions,
        fs_hz=float(config.fs_hz),
        sound_speed_m_s=float(config.sound_speed_m_s),
        fractional_filter_bank=fractional_delay_filter_bank,
    )
    fixed_weights = design_fixed_delay_fractional_weights_from_delay_table(
        delay_table,
        fractional_delay_filter_bank,
        frequencies_hz,
        fs_hz=float(config.fs_hz),
        average_channels=True,
    )
    shading_by_frequency = select_shading_for_frequencies(
        shading_by_channel_bin,
        float(shading_frequency_step_hz),
        frequencies_hz,
    )
    fixed_weights = apply_frequency_shading_to_weights(fixed_weights, shading_by_frequency)

    steering_by_beam = np.stack(
        [
            _arrival_steering(
                positions,
                float(np.rad2deg(np.arctan2(direction[1], direction[0]))),
                frequencies_hz,
                float(config.sound_speed_m_s),
            )
            for direction in beam_directions
        ],
        axis=1,
    )
    source_steering = _arrival_steering(
        positions,
        float(config.source_azimuth_deg),
        frequencies_hz,
        float(config.sound_speed_m_s),
    )
    covariance = _make_source_covariance(
        source_steering,
        source_level_db20=float(config.source_level_db20),
        noise_power_per_channel=float(config.noise_power_per_channel),
    )

    mvdr_designer = LoadedMVDRWeightDesigner(
        diagonal_loading_ratio=float(config.diagonal_loading_ratio)
    )
    mvdr_weights = np.zeros_like(fixed_weights)
    for beam_index in range(fixed_weights.shape[1]):
        # 実運用評価なので、制約は source truth ではなく各 beam の待ち受け方位に置く。
        result = mvdr_designer.compute(
            covariance,
            steering_by_beam[:, beam_index, :],
            fixed_weights[:, beam_index, :],
        )
        mvdr_weights[:, beam_index, :] = result.weights

    rng = np.random.default_rng(int(config.random_seed))
    rows: list[ExternalTapTradeoffRow] = []
    for fir_taps in tap_counts:
        diff_designer = DifferenceCorrectionFIRDesigner(
            fir_taps=int(fir_taps),
            frequencies_hz=frequencies_hz,
            fs_hz=float(config.fs_hz),
        )
        q_rms: list[float] = []
        q_abs: list[float] = []
        target_error: list[float] = []
        weight_error: list[float] = []
        for beam_index in range(fixed_weights.shape[1]):
            diff_result = diff_designer.compute(
                fixed_weights[:, beam_index, :],
                mvdr_weights[:, beam_index, :],
                steering_by_beam[:, beam_index, :],
            )
            q_error = diff_result.diagnostics.q_reconstruction_error
            q_rms.append(float(np.sqrt(np.mean(np.abs(q_error) ** 2))))
            q_abs.append(float(np.max(np.abs(q_error))))
            target_error.append(
                float(
                    np.max(
                        np.abs(
                            diff_result.diagnostics.target_response_final
                            - diff_result.diagnostics.target_response_mvdr
                        )
                    )
                )
            )
            weight_error.append(
                float(
                    np.sqrt(
                        np.mean(
                            np.abs(diff_result.final_weight_freq - mvdr_weights[:, beam_index, :])
                            ** 2
                        )
                    )
                )
            )
        seconds_per_sample = _measure_fir_runtime(
            fir_taps=int(fir_taps),
            n_ch=int(positions.shape[0]),
            sample_count=int(config.benchmark_sample_count),
            repeats=int(config.benchmark_repeats),
            rng=rng,
        )
        rows.append(
            ExternalTapTradeoffRow(
                fir_taps=int(fir_taps),
                frequency_bin_count=int(frequencies_hz.size),
                beam_count=int(fixed_weights.shape[1]),
                channel_count=int(positions.shape[0]),
                mac_per_sample_per_beam=int(positions.shape[0]) * int(fir_taps),
                mac_factor_re_128=float(fir_taps) / 128.0,
                measured_us_per_sample_per_beam=seconds_per_sample * 1.0e6,
                measured_runtime_factor_re_128=0.0,
                max_q_reconstruction_rms_error=max(q_rms),
                max_q_reconstruction_abs_error=max(q_abs),
                max_target_response_abs_error=max(target_error),
                max_final_weight_rms_error=max(weight_error),
            )
        )

    base_runtime = next(
        (row.measured_us_per_sample_per_beam for row in rows if row.fir_taps == 128),
        rows[0].measured_us_per_sample_per_beam,
    )
    normalized_rows: list[ExternalTapTradeoffRow] = []
    for row in rows:
        row_values = {
            **row.__dict__,
            "measured_runtime_factor_re_128": row.measured_us_per_sample_per_beam
            / base_runtime,
        }
        normalized_rows.append(ExternalTapTradeoffRow(**row_values))
    return normalized_rows


def write_tap_tradeoff_outputs(rows: list[ExternalTapTradeoffRow], output_dir: Path) -> None:
    """tap tradeoff の CSV と日本語 Markdown report を保存する。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "external_tap_tradeoff.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].__dict__.keys()))
        writer.writeheader()
        writer.writerows([row.__dict__ for row in rows])

    acceptable_128 = [
        row
        for row in rows
        if row.max_q_reconstruction_rms_error <= 1.0e-5
        and row.max_target_response_abs_error <= 1.0e-5
    ]
    acceptable_192 = [
        row
        for row in rows
        if row.max_q_reconstruction_rms_error <= 1.0e-6
        and row.max_target_response_abs_error <= 1.0e-6
    ]
    lines = [
        "# 外部アレイ係数による差分 FIR tap tradeoff 評価",
        "",
        "## 成果物の定義",
        "",
        "- `external_tap_tradeoff.csv`: tap 数ごとの FIR 化誤差と処理量 metric。",
        "- `mac_per_sample_per_beam`: 1 beam あたりの `n_ch * taps` complex MAC/sample。",

        "- `measured_runtime_factor_re_128`: 128 taps を 1 とした実測処理時間比。",
        "",
        "## 結果",
        "",
        "| taps | runtime re 128 | max q RMS error | max target response error |",
        "|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.fir_taps} | {row.measured_runtime_factor_re_128:.3f} | "
            f"{row.max_q_reconstruction_rms_error:.3e} | "
            f"{row.max_target_response_abs_error:.3e} |"
        )
    lines.extend(["", "## 判断", ""])
    if acceptable_128:
        lines.append(
            f"- `1e-5` 基準を満たす最小 tap 数は `{min(row.fir_taps for row in acceptable_128)}`。"
        )
    if acceptable_192:
        lines.append(
            f"- `1e-6` 基準を満たす最小 tap 数は `{min(row.fir_taps for row in acceptable_192)}`。"
        )
    lines.append(
        "- off-grid source 自己抑圧は tap 数ではなく "
        "steering mismatch 側の問題として別評価する。"
    )
    (output_dir / "external_tap_tradeoff_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def _parse_tap_counts(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def main() -> None:
    """CLI entrypoint。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coe-pos", type=Path, required=True)
    parser.add_argument("--coe-cbfshading", type=Path, required=True)
    parser.add_argument("--shading-df-hz", type=float, default=0.5)
    parser.add_argument("--fractional-delay-npz", type=Path)
    parser.add_argument("--fractional-delay-raw", type=Path)
    parser.add_argument("--fractional-delay-taps", type=int, default=128)
    parser.add_argument("--fractional-delay-frac-min", type=float, default=-0.5)
    parser.add_argument("--fractional-delay-frac-max", type=float, default=0.5)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/beamforming/fixed_delay_diff_mvdr/external_tap_tradeoff"),
    )
    parser.add_argument("--tap-counts", default="16,24,32,48,64,96,128,192,256,384,512")
    args = parser.parse_args()

    positions = load_positions_matlab_raw(args.coe_pos)
    shading = load_complex_shading_matlab_raw(args.coe_cbfshading, n_ch=int(positions.shape[0]))
    if args.fractional_delay_raw is not None:
        filter_bank = load_fractional_delay_filter_bank_matlab_raw(
            args.fractional_delay_raw,
            n_tap=int(args.fractional_delay_taps),
            frac_min=float(args.fractional_delay_frac_min),
            frac_max=float(args.fractional_delay_frac_max),
        )
    elif args.fractional_delay_npz is not None:
        filter_bank = load_fractional_delay_filter_bank_npz(args.fractional_delay_npz)
    else:
        raise ValueError("Specify --fractional-delay-raw or --fractional-delay-npz.")
    rows = evaluate_external_fir_tap_tradeoff(
        array_positions_m=positions,
        shading_by_channel_bin=shading,
        shading_frequency_step_hz=float(args.shading_df_hz),
        fractional_delay_filter_bank=filter_bank,
        tap_counts=_parse_tap_counts(str(args.tap_counts)),
    )
    write_tap_tradeoff_outputs(rows, args.output_dir)
    print(args.output_dir / "external_tap_tradeoff_report.md")


if __name__ == "__main__":
    main()
