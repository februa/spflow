"""運用アレイで同一方位・複数周波数の時間領域固定整相後分離を診断する。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .._validation import require, require_positive_float, require_positive_int
from ..beamforming_evaluation.signal_levels import calculate_tone_projection_rms_level_db20
from .fractional_delay_slc_diagnostics import _run_fractional_delay_diagnostics
from .operational_sparse_array import load_operational_sparse_array
from .time_delay_diagnostics import TimeDelayDiagnosticConfig, TimeDelayDiagnosticSource


@dataclass(frozen=True)
class OperationalSameAzimuthFrequencySeparationConfig:
    """同一方位・複数周波数の固定整相後成分分離条件を保持する。

    このクラスは、運用スパースアレイ定義、保存済み小数遅延 FIR バンク、
    代表処理周波数、同一方位に置く複数 source 周波数、評価時間を保持する。

    入力はアレイファイルと音源条件であり、出力は
    `run_operational_same_azimuth_frequency_separation_diagnostics()` が保存する
    周波数成分別 level / leakage summary である。

    SLC 重み更新、STFT 再合成、MVDR / MUSIC の固有値分解は責務に含めない。
    信号処理上は、時間領域固定整相の出力を単一周波数射影で評価し、
    同一方位の複数周波数を空間分離ではなく周波数成分として扱えるかを見る診断条件に位置づく。
    """

    output_dir: Path
    operational_array_definition_path: Path
    fractional_delay_filter_bank_path: Path
    processing_frequency_hz: float
    source_azimuth_deg: float
    source_frequencies_hz: tuple[float, ...]
    source_levels_db20: tuple[float, ...]
    duration_s: float = 1.0
    noise_level_db20: float = -120.0
    random_seed: int = 1234
    n_beam_az_real: int = 151
    btr_block_size: int = 1024

    def validate(self) -> None:
        """診断条件の境界条件を検証する。

        Raises:
            ValueError: path、周波数、source 数、方位、解析長が不正な場合。

        境界条件:
            周波数成分分離は少なくとも 2 周波数が必要である。
            source level は source 周波数と 1 対 1 に対応するため、長さ不一致を許さない。
        """
        require(Path(self.operational_array_definition_path).exists(), "operational_array_definition_path must exist.")
        require(Path(self.fractional_delay_filter_bank_path).exists(), "fractional_delay_filter_bank_path must exist.")
        require_positive_float("processing_frequency_hz", float(self.processing_frequency_hz))
        require(0.0 <= float(self.source_azimuth_deg) <= 180.0, "source_azimuth_deg must lie in [0, 180].")
        require(len(self.source_frequencies_hz) >= 2, "at least two source frequencies are required.")
        require(len(self.source_frequencies_hz) == len(self.source_levels_db20), "source frequency and level counts must match.")
        for frequency_hz in self.source_frequencies_hz:
            require_positive_float("source_frequency_hz", float(frequency_hz))
        require_positive_float("duration_s", float(self.duration_s))
        require_positive_int("n_beam_az_real", int(self.n_beam_az_real))
        require_positive_int("btr_block_size", int(self.btr_block_size))


def _build_same_azimuth_sources(config: OperationalSameAzimuthFrequencySeparationConfig) -> tuple[TimeDelayDiagnosticSource, ...]:
    """同一方位に置く複数周波数 source を構築する。

    Args:
        config: 周波数成分分離の診断条件。

    Returns:
        `TimeDelayDiagnosticSource` の tuple。長さは `len(source_frequencies_hz)`。
    """
    source_specs: list[TimeDelayDiagnosticSource] = []
    for source_index, (frequency_hz, level_db20) in enumerate(zip(config.source_frequencies_hz, config.source_levels_db20, strict=True)):
        source_specs.append(
            TimeDelayDiagnosticSource(
                azimuth_deg=float(config.source_azimuth_deg),
                frequency_hz=float(frequency_hz),
                level_db20=float(level_db20),
                label=f"F{source_index + 1}",
            )
        )
    return tuple(source_specs)


def _run_fixed_case(
    config: OperationalSameAzimuthFrequencySeparationConfig,
    case_name: str,
    source_specs: tuple[TimeDelayDiagnosticSource, ...],
) -> tuple[dict[str, object], NDArray[np.float64], NDArray[np.float64], float, int, float]:
    """運用 active subset で小数遅延固定整相を 1 ケース実行する。

    Args:
        config: 周波数成分分離の診断条件。
        case_name: 出力ディレクトリ名。
        source_specs: このケースで有効にする source 群。

    Returns:
        `(summary, beam_output, axis_az_deg, fs_hz, active_channel_count, active_aperture_m)`。
        `beam_output` の shape は `[n_beam, n_sample]`、`axis_az_deg` の shape は `[n_beam]`。
    """
    array_definition = load_operational_sparse_array(config.operational_array_definition_path)
    active_indices = array_definition.active_channel_indices_for_frequency(float(config.processing_frequency_hz))

    # active_positions_m shape: [n_active_ch, 3]。
    # 代表処理周波数で使う active subset を固定し、同一方位の複数周波数を同じ時間領域整相器で評価する。
    active_positions_m = np.asarray(array_definition.positions_m[active_indices], dtype=np.float64)
    active_aperture_m = float(np.max(active_positions_m[:, 0]) - np.min(active_positions_m[:, 0]))

    diagnostic_config = TimeDelayDiagnosticConfig(
        output_dir=Path(config.output_dir) / case_name,
        fs_hz=float(array_definition.fs_hz),
        duration_s=float(config.duration_s),
        sound_speed_m_s=float(array_definition.sound_speed_m_s),
        source_specs=source_specs,
        noise_level_db20=float(config.noise_level_db20),
        random_seed=int(config.random_seed),
        array_n_ch=int(active_positions_m.shape[0]),
        array_positions_m=active_positions_m,
        n_beam_az_real=int(config.n_beam_az_real),
        n_beam_az_virtual=0,
        btr_block_size=int(config.btr_block_size),
    )
    summary, _, beam_output, axis_az_deg, _, _ = _run_fractional_delay_diagnostics(
        config=diagnostic_config,
        fractional_delay_filter_bank_path=Path(config.fractional_delay_filter_bank_path),
    )
    return (
        summary,
        np.asarray(beam_output, dtype=np.float64),
        np.asarray(axis_az_deg, dtype=np.float64),
        float(array_definition.fs_hz),
        int(active_indices.size),
        active_aperture_m,
    )


def run_operational_same_azimuth_frequency_separation_diagnostics(
    config: OperationalSameAzimuthFrequencySeparationConfig,
) -> dict[str, object]:
    """運用アレイで同一方位・複数周波数の成分分離を評価する。

    Args:
        config: 運用アレイ、代表処理周波数、同一方位 source 周波数、出力先。

    Returns:
        周波数別 level、off-frequency leakage、BL/FRAZ/BTR summary path を含む summary。

    Raises:
        ValueError: 設定または固定整相の出力 shape が不正な場合。

    境界条件:
        この診断は SLC を適用しない。固定整相後 target beam の時間波形に対し、
        source 周波数ごとの単一 tone 射影で分離可能性を評価する。
    """
    config.validate()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_specs = _build_same_azimuth_sources(config)
    mixed_summary, mixed_beam_output, axis_az_deg, fs_hz, active_channel_count, active_aperture_m = _run_fixed_case(
        config=config,
        case_name="fixed_mixed",
        source_specs=source_specs,
    )
    target_beam_index = int(np.argmin(np.abs(axis_az_deg - float(config.source_azimuth_deg))))
    target_beam_output = mixed_beam_output[target_beam_index, :]
    analysis_bandwidth_hz = 1.0 / float(config.duration_s)

    frequency_levels: list[dict[str, object]] = []
    target_frequency_power_delta_db_values: list[float] = []
    component_leakage_db: list[float] = []
    for source_index, source_spec in enumerate(source_specs):
        source_frequency_hz = float(source_spec.frequency_hz)
        mixed_level_db20 = calculate_tone_projection_rms_level_db20(
            target_beam_output,
            frequency_hz=source_frequency_hz,
            fs_hz=fs_hz,
        )
        target_frequency_power_delta_db = float(mixed_level_db20 - float(source_spec.level_db20))
        target_frequency_power_delta_db_values.append(target_frequency_power_delta_db)

        _, component_beam_output, _, _, _, _ = _run_fixed_case(
            config=config,
            case_name=f"component_{source_index + 1}",
            source_specs=(source_spec,),
        )
        component_target_output = component_beam_output[target_beam_index, :]
        own_level_db20 = calculate_tone_projection_rms_level_db20(
            component_target_output,
            frequency_hz=source_frequency_hz,
            fs_hz=fs_hz,
        )
        off_frequency_levels = [
            calculate_tone_projection_rms_level_db20(
                component_target_output,
                frequency_hz=float(other.frequency_hz),
                fs_hz=fs_hz,
            )
            for other in source_specs
            if float(other.frequency_hz) != source_frequency_hz
        ]
        max_off_frequency_level_db20 = max(off_frequency_levels) if off_frequency_levels else -np.inf
        frequency_bin_leakage_db = float(max_off_frequency_level_db20 - own_level_db20)
        component_leakage_db.append(frequency_bin_leakage_db)

        frequency_levels.append(
            {
                "label": str(source_spec.label),
                "source_frequency_hz": source_frequency_hz,
                "source_level_db20": float(source_spec.level_db20),
                "mixed_target_beam_level_db20": mixed_level_db20,
                "target_frequency_power_delta_db": target_frequency_power_delta_db,
                "component_own_level_db20": own_level_db20,
                "component_max_off_frequency_level_db20": float(max_off_frequency_level_db20),
                "frequency_bin_leakage_db": frequency_bin_leakage_db,
            }
        )

    summary: dict[str, object] = {
        "array_definition_path": str(Path(config.operational_array_definition_path).resolve()),
        "fractional_delay_filter_bank_path": str(Path(config.fractional_delay_filter_bank_path).resolve()),
        "fixed_mixed_summary_path": str((output_dir / "fixed_mixed" / "summary.json").resolve()),
        "level_reference": "dB re input RMS",
        "target_azimuth_deg": float(config.source_azimuth_deg),
        "target_beam_index": int(target_beam_index),
        "target_beam_azimuth_deg": float(axis_az_deg[target_beam_index]),
        "processing_frequency_hz": float(config.processing_frequency_hz),
        "analysis_method": "full-duration single-tone projection on fixed time-domain beam output",
        "analysis_bandwidth_hz": float(analysis_bandwidth_hz),
        "active_channel_count": int(active_channel_count),
        "active_aperture_m": float(active_aperture_m),
        "n_beam": int(mixed_beam_output.shape[0]),
        "n_sample": int(mixed_beam_output.shape[1]),
        "frequency_levels": frequency_levels,
        "max_abs_target_frequency_power_delta_db": float(
            max(abs(value) for value in target_frequency_power_delta_db_values)
        ),
        "worst_frequency_bin_leakage_db": float(max(component_leakage_db)),
        "evaluation_pattern": "slc_same_azimuth_multi_frequency",
        "required_criterion": "frequency_component_separation",
        "mixed_source_metrics": mixed_summary.get("source_metrics", []),
    }
    (output_dir / "same_azimuth_frequency_separation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


__all__ = [
    "OperationalSameAzimuthFrequencySeparationConfig",
    "run_operational_same_azimuth_frequency_separation_diagnostics",
]
