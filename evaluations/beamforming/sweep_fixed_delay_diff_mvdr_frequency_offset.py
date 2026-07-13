"""固定遅延+差分 MVDR の frequency offset sweep を生成する。

このスクリプトは、target と interferer の周波数差を変えながら、
source-preserving scan として両方の source peak が見える条件を調べる。
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from evaluations.beamforming.build_fixed_delay_diff_mvdr_review_pack import (  # noqa: E402
    LEVEL_UNIT_LABEL,
    SourceSpec,
    _arrival_steering,
    _build_array_positions,
    _levels_db,
    _make_covariance,
)
from spflow.beamforming import (  # noqa: E402
    DelayTable,
    DifferenceCorrectionFIRDesigner,
    LoadedMVDRWeightDesigner,
    design_fixed_delay_fractional_weights_from_delay_table,
    design_standard_fractional_delay_filter_bank,
    make_directions,
)
from spflow.beamforming.diagnostic_plotting import require_matplotlib  # noqa: E402

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - 実行環境依存
    plt = None


FloatArray: TypeAlias = NDArray[np.float64]
ComplexArray: TypeAlias = NDArray[np.complex128]

OUTPUT_DIR = ROOT / "artifacts" / "beamforming" / "fixed_delay_diff_mvdr" / "frequency_offset_sweep"
SUMMARY_CSV_PATH = OUTPUT_DIR / "frequency_offset_sweep.csv"
INDEX_MD_PATH = OUTPUT_DIR / "frequency_offset_sweep.md"
PLOT_PATH = OUTPUT_DIR / "frequency_offset_sweep.png"
METHOD_NAME = "diff_mvdr_fir512"
FIR_TAPS = 512
FS_HZ = 32768.0
SOUND_SPEED_M_S = 1500.0
ANALYSIS_DURATION_SEC = 1.0
VISIBILITY_TOLERANCE_DB = -1.0
INTERFERER_AZIMUTH_DEG = 60.0
TARGET_AZIMUTH_DEG = 20.0
OFFSET_HZ_VALUES = (
    0.0,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.0,
    4.0,
    8.0,
    16.0,
    32.0,
    64.0,
    128.0,
)
BASE_FREQUENCY_HZ_VALUES = (1536.0, 4096.0)


@dataclass(frozen=True)
class OffsetSweepRow:
    """frequency offset sweep の 1 行を表す。

    このクラスは、target / interferer の周波数差、各 source peak の保持量、
    protected target beam への leakage reduction、分解可能判定を保持する。

    入力信号生成、MVDR 重み設計、図の保存は責務に含めない。
    信号処理上は、source-preserving scan と local leakage canceller の評価結果を
    CSV へ渡す境界データである。
    """

    base_frequency_hz: float
    offset_hz: float
    target_frequency_hz: float
    interferer_frequency_hz: float
    target_visibility_delta_db: float
    interferer_visibility_delta_db: float
    leakage_reduction_db: float
    target_beam_azimuth_error_deg: float
    interferer_beam_azimuth_error_deg: float
    q_reconstruction_rms_error: float
    analytical_two_frequency_separable: bool
    one_second_bin_separable: bool
    both_source_peaks_visible: bool
    conclusion: str


def _build_frequency_axis(base_frequency_hz: float, interferer_frequency_hz: float) -> FloatArray:
    """評価周波数軸を作る。

    Args:
        base_frequency_hz: target 周波数。単位は Hz。
        interferer_frequency_hz: interferer 周波数。単位は Hz。

    Returns:
        周波数軸。shape は `[n_freq]`、単位は Hz。

    境界条件:
        offset が 128 Hz grid から外れる場合でも、interferer 周波数を明示的に入れる。
        これをしないと nearest bin へ丸められ、0.1 Hz offset の評価が同一周波数扱いになる。
    """
    base_axis = np.arange(768.0, 6144.0 + 1.0, 128.0, dtype=np.float64)
    explicit_points = np.array([base_frequency_hz, interferer_frequency_hz], dtype=np.float64)
    return np.asarray(np.unique(np.concatenate([base_axis, explicit_points])), dtype=np.float64)


def _target_and_interferer_sources(
    base_frequency_hz: float,
    offset_hz: float,
) -> tuple[SourceSpec, SourceSpec]:
    """target と interferer の source 定義を返す。"""
    target = SourceSpec("target", TARGET_AZIMUTH_DEG, base_frequency_hz, 0.0)
    interferer = SourceSpec(
        "interferer",
        INTERFERER_AZIMUTH_DEG,
        base_frequency_hz + offset_hz,
        0.0,
        phase_deg=70.0,
    )
    return target, interferer


def _design_one_beam(
    *,
    fixed_weight: ComplexArray,
    protected_steering: ComplexArray,
    covariance: ComplexArray,
    frequencies_hz: FloatArray,
) -> tuple[ComplexArray, float]:
    """1 beam の diff MVDR FIR512 重みと再構成誤差を返す。

    Args:
        fixed_weight: 固定主経路重み。shape は `[n_freq, n_ch]`。
        protected_steering: 保護 steering。shape は `[n_freq, n_ch]`。
        covariance: 周波数別共分散。shape は `[n_freq, n_ch, n_ch]`。
        frequencies_hz: 設計周波数。shape は `[n_freq]`、単位は Hz。

    Returns:
        `(final_weight, q_reconstruction_rms_error)`。
        `final_weight` の shape は `[n_freq, n_ch]`。
    """
    mvdr_result = LoadedMVDRWeightDesigner(diagonal_loading_ratio=1.0e-2).compute(
        covariance,
        protected_steering,
        fixed_weight,
    )
    diff_result = DifferenceCorrectionFIRDesigner(
        fir_taps=FIR_TAPS,
        frequencies_hz=frequencies_hz,
        fs_hz=FS_HZ,
    ).compute(
        fixed_weight,
        mvdr_result.weights,
        protected_steering,
    )
    # q_reconstruction_error shape: [n_freq, n_ch]。
    # 設計周波数上で差分 FIR が q をどの程度再現したかを RMS で見る。
    q_error = np.asarray(diff_result.diagnostics.q_reconstruction_error, dtype=np.complex128)
    q_reconstruction_rms_error = float(np.sqrt(np.mean(np.abs(q_error) ** 2)))
    return diff_result.final_weight_freq, q_reconstruction_rms_error


def _source_response_level_db(
    weights: ComplexArray,
    source_steering: ComplexArray,
    frequency_index: int,
    beam_index: int,
) -> float:
    """指定 source の指定 beam / 周波数の応答レベルを返す。"""
    response = np.vdot(weights[frequency_index, beam_index, :], source_steering[frequency_index])
    return float(_levels_db(np.asarray(response)))


def _evaluate_offset(
    *,
    base_frequency_hz: float,
    offset_hz: float,
    array_positions_m: FloatArray,
    axis_azimuth_deg: FloatArray,
    beam_directions: FloatArray,
) -> OffsetSweepRow:
    """1 つの frequency offset 条件を評価する。"""
    target, interferer = _target_and_interferer_sources(base_frequency_hz, offset_hz)
    frequencies_hz = _build_frequency_axis(base_frequency_hz, interferer.frequency_hz)
    filter_bank = design_standard_fractional_delay_filter_bank()
    delay_table = DelayTable.from_geometry(
        array_pos_m=array_positions_m,
        dir_cos=beam_directions,
        fs_hz=FS_HZ,
        sound_speed_m_s=SOUND_SPEED_M_S,
        fractional_filter_bank=filter_bank,
    )
    fixed_weights = design_fixed_delay_fractional_weights_from_delay_table(
        delay_table,
        filter_bank,
        frequencies_hz,
        fs_hz=FS_HZ,
        average_channels=True,
    )

    target_beam_index = int(np.argmin(np.abs(axis_azimuth_deg - TARGET_AZIMUTH_DEG)))
    interferer_beam_index = int(np.argmin(np.abs(axis_azimuth_deg - INTERFERER_AZIMUTH_DEG)))
    target_frequency_index = int(np.argmin(np.abs(frequencies_hz - target.frequency_hz)))
    interferer_frequency_index = int(np.argmin(np.abs(frequencies_hz - interferer.frequency_hz)))
    steering_by_source = {
        target.label: _arrival_steering(
            array_positions_m, target, frequencies_hz, sound_speed_m_s=SOUND_SPEED_M_S
        ),
        interferer.label: _arrival_steering(
            array_positions_m, interferer, frequencies_hz, sound_speed_m_s=SOUND_SPEED_M_S
        ),
    }
    all_sources = (target, interferer)
    covariance = np.stack(
        [
            _make_covariance(
                steering_by_source,
                all_sources,
                frequency_index=freq_index,
                noise_power=1.0e-4,
            )
            for freq_index in range(frequencies_hz.size)
        ],
        axis=0,
    )

    target_final_weight, target_q_error = _design_one_beam(
        fixed_weight=fixed_weights[:, target_beam_index, :],
        protected_steering=steering_by_source[target.label],
        covariance=covariance,
        frequencies_hz=frequencies_hz,
    )
    interferer_final_weight, interferer_q_error = _design_one_beam(
        fixed_weight=fixed_weights[:, interferer_beam_index, :],
        protected_steering=steering_by_source[interferer.label],
        covariance=covariance,
        frequencies_hz=frequencies_hz,
    )

    fixed_target_level = _source_response_level_db(
        fixed_weights,
        steering_by_source[target.label],
        target_frequency_index,
        target_beam_index,
    )
    fixed_interferer_level = _source_response_level_db(
        fixed_weights,
        steering_by_source[interferer.label],
        interferer_frequency_index,
        interferer_beam_index,
    )
    final_target_level = float(
        _levels_db(
            np.vdot(
                target_final_weight[target_frequency_index],
                steering_by_source[target.label][target_frequency_index],
            )
        )
    )
    final_interferer_level = float(
        _levels_db(
            np.vdot(
                interferer_final_weight[interferer_frequency_index],
                steering_by_source[interferer.label][interferer_frequency_index],
            )
        )
    )

    fixed_leakage_level = _source_response_level_db(
        fixed_weights,
        steering_by_source[interferer.label],
        interferer_frequency_index,
        target_beam_index,
    )
    final_leakage_level = float(
        _levels_db(
            np.vdot(
                target_final_weight[interferer_frequency_index],
                steering_by_source[interferer.label][interferer_frequency_index],
            )
        )
    )
    target_visibility_delta_db = final_target_level - fixed_target_level
    interferer_visibility_delta_db = final_interferer_level - fixed_interferer_level
    leakage_reduction_db = fixed_leakage_level - final_leakage_level
    analytical_two_frequency_separable = bool(offset_hz > 0.0)
    frequency_bin_width_hz = 1.0 / ANALYSIS_DURATION_SEC
    one_second_bin_separable = bool(abs(offset_hz) >= frequency_bin_width_hz)
    both_source_peaks_visible = bool(
        target_visibility_delta_db >= VISIBILITY_TOLERANCE_DB
        and interferer_visibility_delta_db >= VISIBILITY_TOLERANCE_DB
    )
    if offset_hz == 0.0:
        conclusion = "coherent_same_frequency"
    elif not one_second_bin_separable:
        conclusion = "analytical_only_not_1s_stft"
    elif both_source_peaks_visible:
        conclusion = "separable_and_visible"
    else:
        conclusion = "not_visible"

    return OffsetSweepRow(
        base_frequency_hz=float(base_frequency_hz),
        offset_hz=float(offset_hz),
        target_frequency_hz=float(target.frequency_hz),
        interferer_frequency_hz=float(interferer.frequency_hz),
        target_visibility_delta_db=float(target_visibility_delta_db),
        interferer_visibility_delta_db=float(interferer_visibility_delta_db),
        leakage_reduction_db=float(leakage_reduction_db),
        target_beam_azimuth_error_deg=abs(
            float(axis_azimuth_deg[target_beam_index]) - TARGET_AZIMUTH_DEG
        ),
        interferer_beam_azimuth_error_deg=abs(
            float(axis_azimuth_deg[interferer_beam_index]) - INTERFERER_AZIMUTH_DEG
        ),
        q_reconstruction_rms_error=float(max(target_q_error, interferer_q_error)),
        analytical_two_frequency_separable=analytical_two_frequency_separable,
        one_second_bin_separable=one_second_bin_separable,
        both_source_peaks_visible=both_source_peaks_visible,
        conclusion=conclusion,
    )


def _write_csv(rows: list[OffsetSweepRow]) -> None:
    """sweep 結果を CSV 保存する。"""
    SUMMARY_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = tuple(OffsetSweepRow.__dataclass_fields__.keys())
    with SUMMARY_CSV_PATH.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: getattr(row, field) for field in fieldnames})


def _write_plot(rows: list[OffsetSweepRow]) -> None:
    """offset sweep の可視化を保存する。"""
    require_matplotlib()
    if plt is None:
        raise RuntimeError("matplotlib is required to build sweep plot.")
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharey=True)
    for axis, base_frequency_hz in zip(axes, BASE_FREQUENCY_HZ_VALUES, strict=True):
        target_rows = [row for row in rows if row.base_frequency_hz == float(base_frequency_hz)]
        offsets = np.array([max(row.offset_hz, 0.01) for row in target_rows], dtype=np.float64)
        target_delta = np.array([row.target_visibility_delta_db for row in target_rows])
        interferer_delta = np.array([row.interferer_visibility_delta_db for row in target_rows])
        axis.plot(offsets, target_delta, marker="o", label="target peak")
        axis.plot(offsets, interferer_delta, marker="s", label="interferer peak")
        axis.axhline(VISIBILITY_TOLERANCE_DB, color="black", linewidth=0.8, linestyle="--")
        axis.axvline(1.0 / ANALYSIS_DURATION_SEC, color="tab:red", linewidth=0.8, linestyle=":")
        axis.set_xscale("log")
        axis.set_title(f"target {base_frequency_hz:.1f} Hz")
        axis.set_xlabel("Frequency offset [Hz]")
        axis.grid(True, alpha=0.25)
    axes[0].set_ylabel(f"Peak level delta [{LEVEL_UNIT_LABEL} re fixed]")
    axes[0].legend(loc="best")
    fig.suptitle("diff MVDR FIR512 source visibility vs frequency offset")
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_index(rows: list[OffsetSweepRow]) -> None:
    """sweep 結果の Markdown index を保存する。"""
    first_separable_by_base: dict[float, float | None] = {}
    for base_frequency_hz in BASE_FREQUENCY_HZ_VALUES:
        candidates = [
            row.offset_hz
            for row in rows
            if row.base_frequency_hz == float(base_frequency_hz)
            and row.conclusion == "separable_and_visible"
        ]
        first_separable_by_base[float(base_frequency_hz)] = min(candidates) if candidates else None

    lines = [
        "# Fixed Delay + Difference MVDR Frequency Offset Sweep",
        "",
        "## 評価条件",
        "",
        f"- method: `{METHOD_NAME}`",
        f"- FIR taps: `{FIR_TAPS}`",
        f"- visibility tolerance: `{VISIBILITY_TOLERANCE_DB:.1f} dB re fixed source peak`",
        f"- 1 秒観測の周波数 bin 幅: `{1.0 / ANALYSIS_DURATION_SEC:.1f} Hz`",
        f"- CSV: `{SUMMARY_CSV_PATH.name}`",
        f"- plot: `{PLOT_PATH.name}`",
        "",
        "## 結論",
        "",
    ]
    for base_frequency_hz, first_offset_hz in first_separable_by_base.items():
        if first_offset_hz is None:
            lines.append(f"- target {base_frequency_hz:.1f} Hz: sweep 範囲では未分解。")
        else:
            lines.append(
                f"- target {base_frequency_hz:.1f} Hz: 1 秒観測基準では "
                f"{first_offset_hz:.2f} Hz 以上で分解可能。"
            )
    lines.extend(
        [
            "",
            "0.1 Hz は解析周波数軸に明示すれば数式上は別周波数として扱えるが、",
            "1 秒観測の STFT bin 幅では分解可能とはみなさない。",
            "",
            "## Rows",
            "",
            (
                "| base Hz | offset Hz | target delta | interferer delta | "
                "leakage reduction | conclusion |"
            ),
            "|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            f"{row.base_frequency_hz:.1f} | "
            f"{row.offset_hz:.2f} | "
            f"{row.target_visibility_delta_db:.3f} | "
            f"{row.interferer_visibility_delta_db:.3f} | "
            f"{row.leakage_reduction_db:.3f} | "
            f"`{row.conclusion}` |"
        )
    INDEX_MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def build_frequency_offset_sweep() -> None:
    """frequency offset sweep の成果物を生成する。"""
    array_positions = _build_array_positions(n_ch=32, spacing_m=0.05)
    directions, axis_azimuth_deg, _ = make_directions(
        az_min_deg=0.0,
        az_max_deg=180.0,
        el_min_deg=0.0,
        el_max_deg=0.0,
        n_beam_az_real=121,
        n_beam_az_virtual=0,
        n_beam_el=1,
        array_side="right side",
        el_preset_deg=[0.0],
    )
    beam_directions = directions.T.astype(np.float64)
    rows: list[OffsetSweepRow] = []
    for base_frequency_hz in BASE_FREQUENCY_HZ_VALUES:
        for offset_hz in OFFSET_HZ_VALUES:
            rows.append(
                _evaluate_offset(
                    base_frequency_hz=float(base_frequency_hz),
                    offset_hz=float(offset_hz),
                    array_positions_m=array_positions,
                    axis_azimuth_deg=axis_azimuth_deg.astype(np.float64),
                    beam_directions=beam_directions,
                )
            )
    _write_csv(rows)
    _write_plot(rows)
    _write_index(rows)


def main() -> None:
    """CLI entrypoint。"""
    build_frequency_offset_sweep()
    print(f"saved sweep csv to {SUMMARY_CSV_PATH}")
    print(f"saved sweep index to {INDEX_MD_PATH}")
    print(f"saved sweep plot to {PLOT_PATH}")


if __name__ == "__main__":
    main()
