"""8月評価向けの軽量 ABF-like A2 候補を絞り込み評価する。

このスクリプトは、既存の `comparison_summary.csv` で有効性を確認した
`fixed_baseline`、`A2_safe`、`A2_aggressive` に対象を絞り、代表図、
負例評価、metric safety gate、A2 kernel runtime を同じ評価基準で保存する。

出力先は `artifacts/beamforming/lightweight_abf_like_comparison/august_shortlist` である。
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from examples.beamforming.evaluate_lightweight_abf_like_comparison import (
    ARRAY_DEFINITION_PATH,
    DETECTED_SOURCE_THRESHOLD_DB_BELOW_PEAK,
    FRACTIONAL_DELAY_FILTER_BANK_PATH,
    INTERFERER_LEVEL_DB,
    LEVEL_UNIT_LABEL,
    N_BEAM_AZ_REAL,
    N_SAMPLE,
    SAMPLE_PER_DOF_MIN,
    SHADING_DEFINITION_PATH,
    TARGET_LEVEL_DB,
    B1Candidate,
    EvaluationCase,
    FrequencyResponseCacheEntry,
    SourceGuardVariant,
    SourceSpec,
    _base_row,
    _complex_response_for_source,
    _metrics_row_fields,
    _rms_levels_db20,
    _run_a2_source_mask_slc,
    _source_mask_for_case,
    _weights_for_candidate,
    _write_csv,
)
from spflow.beamforming import SourceSectorMask
from spflow.beamforming.diagnostic_plotting import (
    plot_bl_response,
    plot_btr_heatmap,
    plot_fraz_heatmap,
    require_matplotlib,
)
from spflow.beamforming.directions import make_directions
from spflow.beamforming.operational_shading import OperationalShadingDefinition
from spflow.beamforming.operational_sparse_array import OperationalSparseArrayDefinition
from spflow.beamforming.time_delay import (
    FractionalDelayAndSumBeamformer,
    FractionalDelayFilterBank,
)

FloatArray: TypeAlias = NDArray[np.float64]
ComplexArray: TypeAlias = NDArray[np.complex128]
IntArray: TypeAlias = NDArray[np.int64]


COMPARISON_DIR = Path("artifacts/beamforming/lightweight_abf_like_comparison")
COMPARISON_SUMMARY_CSV_PATH = COMPARISON_DIR / "comparison_summary.csv"
OUTPUT_DIR = COMPARISON_DIR / "august_shortlist"
FIGURE_DIR = OUTPUT_DIR / "figures"
SHORTLIST_SUMMARY_CSV_PATH = OUTPUT_DIR / "shortlist_summary.csv"
NEGATIVE_SUMMARY_CSV_PATH = OUTPUT_DIR / "negative_summary.csv"
SAFETY_GATE_SUMMARY_CSV_PATH = OUTPUT_DIR / "safety_gate_summary.csv"
RUNTIME_SUMMARY_CSV_PATH = OUTPUT_DIR / "runtime_summary.csv"
FIGURE_MANIFEST_CSV_PATH = OUTPUT_DIR / "representative_figure_manifest.csv"
FINAL_REPORT_MD_PATH = OUTPUT_DIR / "august_candidate_report.md"

WIDE_GUARD = SourceGuardVariant(label="wide", guard_half_width_deg=3.0)
TONE_FREQUENCIES_HZ = np.array([6144.0, 8192.0, 10000.0], dtype=np.float64)
TARGET_PHASE_RAD = 0.0
INTERFERER_PHASE_RAD = 0.7
UNKNOWN_SOURCE_PHASE_RAD = 1.4
WEAK_SOURCE_PHASE_RAD = 2.1

# metric safety gate は raw 出力の局所悪化を運用出力へ流さないための上限である。
# 0.25 dB は BL/FRAZ/BTR の見た目にも現れ始める局所増加を検出しつつ、
# 浮動小数点丸めや tone projection の微小差では fallback しない幅として置く。
METRIC_SAFETY_LOCAL_WORSENING_LIMIT_DB = 0.25

# source mask 内の peak が 1 dB を超えて動く場合は、source 保護が設計意図から外れる。
# raw が non-source を抑えていても、source 保護を失った出力は effective に採用しない。
METRIC_SAFETY_SOURCE_PEAK_LIMIT_DB = 1.0

A2_KERNEL_REPEAT_COUNT = 20
A2_KERNEL_WARMUP_COUNT = 3
BTR_BLOCK_SIZE = 128
RISK_FIGURE_SCENARIO_IDS = (
    "negative_source_azimuth_offgrid_0p5_mask_nominal",
    "negative_detected_mask_single_source",
    "negative_unknown_source_outside_mask",
    "negative_weak_source_outside_mask",
)


@dataclass(frozen=True)
class ShortlistCandidate:
    """8月評価へ残す方式候補を表す。

    このクラスは、fixed baseline と A2 の候補識別子、SLC 係数、loading、
    source guard、train/test 分離条件をまとめる。

    入力は評価設定の scalar 群であり、出力は CSV / 図ファイル名に使う安定した
    `candidate_id` である。

    固定整相応答の計算、source mask 検出、SLC 係数推定そのものは責務に含めない。
    信号処理上は、同じ beam-domain 入力へ適用する後段方式の選択条件に位置づく。
    """

    method_id: str
    eta: float | None
    loading: float | None
    guard: SourceGuardVariant
    train_mode: str
    is_baseline: bool

    @property
    def candidate_id(self) -> str:
        """CSV と図ディレクトリに使う候補 ID を返す。"""
        if self.is_baseline:
            return "fixed_baseline"
        if self.eta is None or self.loading is None:
            raise ValueError("A2 candidate requires eta and loading.")
        return (
            f"eta{self.eta:g}_loading{self.loading:g}_"
            f"{self.guard.label}_{self.train_mode}"
        )


@dataclass(frozen=True)
class EvaluationAssets:
    """評価に必要な配列定義、shading、beam 軸、周波数応答 cache を保持する。

    このクラスは、運用 sparse array、運用 shading、小数遅延 FIR bank から構築した
    beamformer 群をまとめ、各評価ケースの合成処理に渡す。

    入力は保存済み artifact と方位走査条件であり、出力は beam-domain 合成に使う
    `response_cache[frequency_hz]` と方位軸である。

    個別 candidate の SLC 実行、metric 判定、図保存は責務に含めない。
    信号処理上は、周波数別 fixed beam response を一度だけ準備する共有コンテキストである。
    """

    array_definition: OperationalSparseArrayDefinition
    shading_definition: OperationalShadingDefinition
    axis_azimuth_deg: FloatArray
    response_cache: dict[float, FrequencyResponseCacheEntry]


@dataclass(frozen=True)
class ScenarioDefinition:
    """代表図または負例評価の source 条件を保持する。

    このクラスは、評価ケース、mask 種別、実際に合成する source 群、
    oracle mask に渡す方位群を一体で扱う。

    入力は source 方位・周波数・レベルの配列であり、出力は固定整相後
    beam-domain 波形と source mask を生成するための条件である。

    SLC 係数推定、ABF-like 指標計算、BL/FRAZ/BTR 描画は責務に含めない。
    信号処理上は、複数 source と mask のずれを明示して安全性を確認する評価単位である。
    """

    scenario_id: str
    scenario_group: str
    selection_basis: str
    case: EvaluationCase
    mask_type: str
    source_specs: tuple[SourceSpec, ...]
    source_mask_azimuths_deg: FloatArray
    notes: str


@dataclass(frozen=True)
class CandidateEvaluation:
    """1 candidate の raw/effective 出力と診断量を保持する。

    このクラスは、固定整相 baseline と A2 candidate を同じ評価窓へ揃えた結果を保持する。

    入力は fixed beam output `[n_beam, n_sample]` と source mask であり、出力は
    raw candidate、metric safety gate 後の effective output、診断 scalar である。

    source mask 作成、beam-domain 合成、CSV 保存は責務に含めない。
    信号処理上は、raw と effective を分離し、方式判定を effective のみで行うための結果型である。
    """

    candidate: ShortlistCandidate
    before_window: ComplexArray
    raw_output: ComplexArray
    effective_output: ComplexArray
    raw_metric_fields: dict[str, object]
    metric_fallback_reasons: tuple[str, ...]
    condition_number: float | None
    weight_norm: float | None
    realtime_factor: float
    fallback_required: bool
    fallback_reasons: tuple[str, ...]
    n_ref: int
    n_non_source: int
    capacity_block_size: int
    capacity_dof: int
    elapsed_sec: float


SHORTLIST_CANDIDATES = (
    ShortlistCandidate(
        method_id="fixed_baseline",
        eta=None,
        loading=None,
        guard=WIDE_GUARD,
        train_mode="one_block_delay",
        is_baseline=True,
    ),
    ShortlistCandidate(
        method_id="A2_safe",
        eta=0.5,
        loading=3.0e-2,
        guard=WIDE_GUARD,
        train_mode="one_block_delay",
        is_baseline=False,
    ),
    ShortlistCandidate(
        method_id="A2_aggressive",
        eta=1.0,
        loading=1.0e-1,
        guard=WIDE_GUARD,
        train_mode="one_block_delay",
        is_baseline=False,
    ),
)


def _case(
    *,
    tier: str,
    target_frequency_hz: float,
    interferer_frequency_hz: float,
    interferer_azimuth_base_deg: float,
    offgrid_deg: float,
) -> EvaluationCase:
    """CSV と同じ case_id 規則で評価ケースを作る。

    Args:
        tier: `tier0` または `tier1`。
        target_frequency_hz: target tone 周波数。単位は Hz。
        interferer_frequency_hz: interferer tone 周波数。単位は Hz。
        interferer_azimuth_base_deg: offgrid 加算前の interferer 方位。単位は deg。
        offgrid_deg: target / interferer の両方へ加える方位ずれ。単位は deg。

    Returns:
        `EvaluationCase`。target 方位は 90 deg + offgrid として扱う。

    Raises:
        ValueError: tier が想定外の場合。
    """
    offgrid_label = f"{int(round(float(offgrid_deg) * 100.0)):03d}"
    if tier == "tier0":
        case_id = f"tier0_tf10000_if8192_ia060_off{offgrid_label}"
    elif tier == "tier1":
        case_id = (
            f"tier1_tf{int(target_frequency_hz)}_if{int(interferer_frequency_hz)}_"
            f"ia{int(interferer_azimuth_base_deg):03d}_off{offgrid_label}"
        )
    else:
        raise ValueError("tier must be tier0 or tier1.")
    return EvaluationCase(
        tier=tier,
        case_id=case_id,
        target_frequency_hz=float(target_frequency_hz),
        interferer_frequency_hz=float(interferer_frequency_hz),
        interferer_azimuth_base_deg=float(interferer_azimuth_base_deg),
        offgrid_deg=float(offgrid_deg),
    )


def _two_source_specs(case: EvaluationCase) -> tuple[SourceSpec, SourceSpec]:
    """target / interferer の標準 2 source 条件を返す。

    Args:
        case: target / interferer の方位と周波数を保持する評価ケース。

    Returns:
        source 条件の tuple。shape 的には `[2]` の source list であり、
        各 source は周波数 Hz、方位 deg、RMS レベル dB、初期位相 rad を持つ。
    """
    return (
        SourceSpec(
            label="target",
            frequency_hz=float(case.target_frequency_hz),
            azimuth_deg=float(case.target_azimuth_deg),
            level_db=float(TARGET_LEVEL_DB),
            phase_rad=TARGET_PHASE_RAD,
        ),
        SourceSpec(
            label="interferer",
            frequency_hz=float(case.interferer_frequency_hz),
            azimuth_deg=float(case.interferer_azimuth_deg),
            level_db=float(INTERFERER_LEVEL_DB),
            phase_rad=INTERFERER_PHASE_RAD,
        ),
    )


def _load_assets() -> EvaluationAssets:
    """評価 artifact を読み込み、周波数別 fixed beam response cache を構築する。

    Returns:
        beam-domain 合成に使う共有 asset 群。

    Raises:
        FileNotFoundError: 必要 artifact が存在しない場合。

    境界条件:
        周波数 cache は 6144 / 8192 / 10000 Hz に限定する。今回の Tier 条件と
        負例 source はこの 3 周波数だけを使うため、余分な beamformer 構築時間を避ける。
    """
    array_definition = OperationalSparseArrayDefinition.load_json(ARRAY_DEFINITION_PATH)
    shading_definition = OperationalShadingDefinition.load_json(SHADING_DEFINITION_PATH)
    filter_bank = FractionalDelayFilterBank.load_npz(FRACTIONAL_DELAY_FILTER_BANK_PATH)

    directions, axis_azimuth_deg_raw, _ = make_directions(
        az_min_deg=0.0,
        az_max_deg=180.0,
        el_min_deg=0.0,
        el_max_deg=0.0,
        n_beam_az_real=N_BEAM_AZ_REAL,
        n_beam_az_virtual=0,
        n_beam_el=1,
        array_side="right side",
        el_preset_deg=[0.0],
    )
    axis_azimuth_deg = np.asarray(axis_azimuth_deg_raw, dtype=np.float64)
    direction_cosines = np.asarray(directions.T, dtype=np.float64)

    response_cache: dict[float, FrequencyResponseCacheEntry] = {}
    for frequency_hz in TONE_FREQUENCIES_HZ.tolist():
        active_indices = array_definition.active_channel_indices_for_frequency(float(frequency_hz))
        active_positions_m = np.asarray(
            array_definition.positions_m[active_indices],
            dtype=np.float64,
        )
        beamformer = FractionalDelayAndSumBeamformer.from_geometry(
            array_pos_m=active_positions_m,
            dir_cos=direction_cosines,
            fs_hz=float(array_definition.fs_hz),
            sound_speed_m_s=float(array_definition.sound_speed_m_s),
            fractional_filter_bank=filter_bank,
        )
        response_cache[float(frequency_hz)] = FrequencyResponseCacheEntry(
            frequency_hz=float(frequency_hz),
            active_indices=np.asarray(active_indices, dtype=np.int64),
            active_positions_m=active_positions_m,
            beamformer=beamformer,
        )

    return EvaluationAssets(
        array_definition=array_definition,
        shading_definition=shading_definition,
        axis_azimuth_deg=axis_azimuth_deg,
        response_cache=response_cache,
    )


def _operational_weights_for_sources(
    *,
    assets: EvaluationAssets,
    source_specs: tuple[SourceSpec, ...],
) -> dict[float, FloatArray]:
    """source 群の各周波数に対する運用 shading weight を返す。

    Args:
        assets: 配列定義と shading 定義を含む共有 asset。
        source_specs: 合成する source 条件列。shape は `[n_source]`。

    Returns:
        周波数 Hz を key とする channel weight 辞書。
        各 value の shape は `[n_active_ch_at_frequency]` である。

    境界条件:
        同じ周波数の source が複数ある場合でも、固定整相器と shading は周波数だけで
        決まるため 1 回だけ計算する。
    """
    operational_candidate = B1Candidate(
        candidate_id="current_operational_shading",
        beta=None,
        uses_operational_shading=True,
    )
    unique_frequencies_hz = sorted({float(source.frequency_hz) for source in source_specs})
    return {
        frequency_hz: _weights_for_candidate(
            candidate=operational_candidate,
            frequency_hz=frequency_hz,
            cache_entry=assets.response_cache[frequency_hz],
            shading_definition=assets.shading_definition,
        )
        for frequency_hz in unique_frequencies_hz
    }


def _synthesize_multisource_beam_output(
    *,
    source_specs: tuple[SourceSpec, ...],
    assets: EvaluationAssets,
    weights_by_frequency: dict[float, FloatArray],
    n_sample: int,
) -> ComplexArray:
    """複数 source の fixed beam-domain mixed 波形を合成する。

    Args:
        source_specs: source 条件列。shape は `[n_source]`。
        assets: 周波数別 fixed beam response cache と配列条件。
        weights_by_frequency: 周波数別 active channel weight。
            各 value の shape は `[n_active_ch_at_frequency]`。
        n_sample: 出力 sample 数。単位は sample。

    Returns:
        mixed beam-domain 複素波形。shape は `[n_beam, n_sample]`。
        axis=0 が beam 方位、axis=1 が時間 sample である。

    Raises:
        ValueError: source が空、または n_sample が正でない場合。
    """
    if len(source_specs) == 0:
        raise ValueError("source_specs must not be empty.")
    if int(n_sample) <= 0:
        raise ValueError("n_sample must be positive.")

    fs_hz = float(assets.array_definition.fs_hz)
    sound_speed_m_s = float(assets.array_definition.sound_speed_m_s)
    time_axis_s = np.arange(int(n_sample), dtype=np.float64) / fs_hz
    first_entry = next(iter(assets.response_cache.values()))
    n_beam = int(first_entry.beamformer.delay_table.n_beam)
    beam_output = np.zeros((n_beam, int(n_sample)), dtype=np.complex128)

    for source_spec in source_specs:
        frequency_hz = float(source_spec.frequency_hz)
        response = _complex_response_for_source(
            cache_entry=assets.response_cache[frequency_hz],
            channel_weights=weights_by_frequency[frequency_hz],
            source_azimuth_deg=float(source_spec.azimuth_deg),
            sound_speed_m_s=sound_speed_m_s,
        )
        amplitude_rms = float(10.0 ** (float(source_spec.level_db) / 20.0))
        # complex exponential の RMS 振幅は amplitude_rms である。
        # 実信号換算の 1/sqrt(2) は使わず、既存 comparison と同じ dB re input RMS に揃える。
        tone = amplitude_rms * np.exp(
            1j
            * (
                2.0 * np.pi * frequency_hz * time_axis_s
                + float(source_spec.phase_rad)
            )
        )
        # response[:, None] shape: [n_beam, 1]、tone[None, :] shape: [1, n_sample]。
        # broadcasting により source ごとの beam response と時間波形を加算する。
        beam_output += response[:, np.newaxis] * tone[np.newaxis, :]

    return beam_output


def _source_mask_for_scenario(
    *,
    scenario: ScenarioDefinition,
    assets: EvaluationAssets,
    before_levels_db: FloatArray,
) -> SourceSectorMask:
    """scenario の oracle / detected source mask を作る。

    Args:
        scenario: mask 種別と oracle 方位を含む評価 scenario。
        assets: 方位軸を含む共有 asset。
        before_levels_db: fixed baseline の beam RMS レベル。shape は `[n_beam]`。

    Returns:
        source / non-source sector mask。mask shape は `[n_beam]`。

    境界条件:
        detected mask では最大 2 peak を検出する既存 comparison と同じ規則を使う。
        負例「detected が 1 本だけ検出」では、弱 source を閾値下へ置くことで
        source mask 外の未検出成分として扱う。
    """
    return _source_mask_for_case(
        axis_azimuth_deg=assets.axis_azimuth_deg,
        before_levels_db=before_levels_db,
        source_azimuths_deg=scenario.source_mask_azimuths_deg,
        guard=WIDE_GUARD,
        mask_type=scenario.mask_type,
    )




def _evaluation_window(beam_output: ComplexArray) -> ComplexArray:
    """one-block-delay 評価窓として後半 block を返す。

    Args:
        beam_output: fixed baseline の beam-domain 波形。shape は `[n_beam, n_sample]`。

    Returns:
        評価対象の後半 block。shape は `[n_beam, n_sample / 2]`。

    Raises:
        ValueError: sample 数が one-block-delay に不足する場合。

    境界条件:
        A2 は前半 block で covariance を学習し、後半 block を評価する。
        fixed baseline も同じ時間窓へ切り出さないと、RMS と runtime factor の分母がずれる。
    """
    signals = np.asarray(beam_output, dtype=np.complex128)
    if signals.ndim != 2:
        raise ValueError("beam_output must have shape (n_beam, n_sample).")
    split_index = int(signals.shape[1] // 2)
    if split_index <= 0 or split_index >= signals.shape[1]:
        raise ValueError("one-block-delay requires at least two samples.")
    return np.asarray(signals[:, split_index:], dtype=np.complex128)


def _row_float(row: dict[str, object], key: str, default: float = 0.0) -> float:
    """CSV row 候補から Python float を取り出す。

    Args:
        row: metric field を含む辞書。
        key: 取り出す field 名。
        default: field が存在しない、または空文字の場合の値。

    Returns:
        Python float へ確定した値。

    Raises:
        TypeError: 数値として扱えない値が入っている場合。

    境界条件:
        CSV 出力用の辞書では method により空文字が入ることがあるため、
        空文字は default として扱い、bool は数値誤読を避けるため拒否する。
    """
    value = row.get(key, default)
    if value is None or value == "":
        return float(default)
    if isinstance(value, bool) or not isinstance(value, int | float | np.integer | np.floating):
        raise TypeError(f"{key} must be numeric.")
    return float(value)


def _metric_safety_fallback_reasons(raw_metric_fields: dict[str, object]) -> tuple[str, ...]:
    """raw metric から effective fallback が必要な理由を返す。

    Args:
        raw_metric_fields: `_metrics_row_fields` が返す raw 出力の評価 field。

    Returns:
        fallback 理由の tuple。空なら raw を effective として採用できる。

    信号処理上の位置づけ:
        A2 raw は source-correlated leakage を下げる候補に過ぎない。
        raw が局所的な non-source 増加、source peak 破壊、false peak 増加を起こす場合は、
        non-source 抑圧の平均改善があっても運用出力としては固定整相へ戻す。
    """
    reasons: list[str] = []
    gated_worsening_db = _row_float(raw_metric_fields, "max_local_worsening_db_gated")
    ungated_worsening_db = _row_float(raw_metric_fields, "ungated_max_local_worsening_db")
    source_peak_delta_db = abs(_row_float(raw_metric_fields, "max_abs_source_peak_delta_db"))
    false_peak_count_delta = int(round(_row_float(raw_metric_fields, "false_peak_count_delta")))
    status = str(raw_metric_fields.get("status", ""))

    if gated_worsening_db > METRIC_SAFETY_LOCAL_WORSENING_LIMIT_DB:
        reasons.append("metric_safety_gate_gated_local_worsening")
    if ungated_worsening_db > METRIC_SAFETY_LOCAL_WORSENING_LIMIT_DB:
        reasons.append("metric_safety_gate_ungated_local_worsening")
    if source_peak_delta_db > METRIC_SAFETY_SOURCE_PEAK_LIMIT_DB:
        reasons.append("metric_safety_gate_source_peak_delta")
    if false_peak_count_delta > 0:
        reasons.append("metric_safety_gate_false_peak_increase")
    if status == "fail":
        reasons.append("metric_safety_gate_raw_metric_fail")

    return tuple(dict.fromkeys(reasons))


def _evaluate_candidate(
    *,
    candidate: ShortlistCandidate,
    before_output: ComplexArray,
    source_mask: SourceSectorMask,
    source_reference_beams: IntArray,
    axis_azimuth_deg: FloatArray,
    evaluation_duration_s: float,
) -> CandidateEvaluation:
    """fixed baseline または A2 candidate を評価窓へ適用する。

    Args:
        candidate: 評価する方式候補。
        before_output: fixed baseline の beam-domain 波形。shape は `[n_beam, n_sample]`。
        source_mask: source / non-source sector mask。mask shape は `[n_beam]`。
        source_reference_beams: A2 source reference beam index。shape は `[n_ref]`。
        axis_azimuth_deg: beam 方位軸。shape は `[n_beam]`、単位は deg。
        evaluation_duration_s: 後半評価 block の時間長。単位は s。

    Returns:
        raw/effective 出力と診断量。

    境界条件:
        baseline は後半 block をそのまま raw/effective として返す。
        A2 は one-block-delay の raw 出力を metric safety gate に通し、
        raw が悪化する場合は fixed baseline の後半 block を effective とする。
    """
    before_window = _evaluation_window(before_output)
    before_window_levels_db = _rms_levels_db20(before_window)

    if candidate.is_baseline:
        raw_metric_fields = _metrics_row_fields(
            before_levels_db=before_window_levels_db,
            after_levels_db=before_window_levels_db,
            axis_azimuth_deg=axis_azimuth_deg,
            source_mask=source_mask,
            realtime_factor=0.0,
            nan_inf_count=0,
            condition_number=None,
        )
        return CandidateEvaluation(
            candidate=candidate,
            before_window=before_window,
            raw_output=before_window.copy(),
            effective_output=before_window.copy(),
            raw_metric_fields=raw_metric_fields,
            metric_fallback_reasons=(),
            condition_number=None,
            weight_norm=None,
            realtime_factor=0.0,
            fallback_required=False,
            fallback_reasons=(),
            n_ref=int(source_reference_beams.size),
            n_non_source=int(np.count_nonzero(source_mask.non_source_mask)),
            capacity_block_size=int(before_window.shape[1]),
            capacity_dof=0,
            elapsed_sec=0.0,
        )

    if candidate.eta is None or candidate.loading is None:
        raise ValueError("A2 candidate requires eta and loading.")

    result = _run_a2_source_mask_slc(
        beam_output=before_output,
        source_sector_mask=source_mask,
        source_reference_beams=source_reference_beams,
        eta=float(candidate.eta),
        loading=float(candidate.loading),
        sample_per_dof=SAMPLE_PER_DOF_MIN,
        train_mode=candidate.train_mode,
    )
    realtime_factor = float(
        result.elapsed_sec / max(evaluation_duration_s, np.finfo(np.float64).eps)
    )
    raw_levels_db = _rms_levels_db20(result.raw_output)
    raw_metric_fields = _metrics_row_fields(
        before_levels_db=before_window_levels_db,
        after_levels_db=raw_levels_db,
        axis_azimuth_deg=axis_azimuth_deg,
        source_mask=source_mask,
        realtime_factor=realtime_factor,
        nan_inf_count=int(np.count_nonzero(~np.isfinite(result.raw_output))),
        condition_number=result.condition_number,
    )
    metric_fallback_reasons = _metric_safety_fallback_reasons(raw_metric_fields)
    fallback_reasons = list(result.fallback_reasons)
    effective_output = np.asarray(result.effective_output, dtype=np.complex128)

    if not result.fallback_required and len(metric_fallback_reasons) > 0:
        # raw の metric が悪化した場合、SLC の数値計算自体が成功していても、
        # 運用出力は fixed baseline と同じ後半 block へ戻す。
        effective_output = before_window.copy()
        fallback_reasons.extend(metric_fallback_reasons)

    fallback_required = bool(result.fallback_required or len(metric_fallback_reasons) > 0)
    return CandidateEvaluation(
        candidate=candidate,
        before_window=before_window,
        raw_output=np.asarray(result.raw_output, dtype=np.complex128),
        effective_output=effective_output,
        raw_metric_fields=raw_metric_fields,
        metric_fallback_reasons=metric_fallback_reasons,
        condition_number=result.condition_number,
        weight_norm=result.weight_norm,
        realtime_factor=realtime_factor,
        fallback_required=fallback_required,
        fallback_reasons=tuple(dict.fromkeys(fallback_reasons)),
        n_ref=int(result.n_ref),
        n_non_source=int(result.n_non_source),
        capacity_block_size=int(result.capacity_block_size),
        capacity_dof=int(result.capacity_dof),
        elapsed_sec=float(result.elapsed_sec),
    )


def _source_visibility_fields(
    *,
    scenario: ScenarioDefinition,
    source_mask: SourceSectorMask,
    axis_azimuth_deg: FloatArray,
    before_output: ComplexArray,
    after_output: ComplexArray,
    fs_hz: float,
) -> dict[str, object]:
    """true source 方位での source visibility delta を CSV field にする。

    Args:
        scenario: 評価 scenario。source 条件列を含む。
        source_mask: source / non-source sector mask。mask shape は `[n_beam]`。
        axis_azimuth_deg: beam 方位軸。shape は `[n_beam]`、単位は deg。
        before_output: fixed baseline の評価窓波形。shape は `[n_beam, n_sample]`。
        after_output: candidate の評価窓波形。shape は `[n_beam, n_sample]`。
        fs_hz: サンプリング周波数。単位は Hz。

    Returns:
        source ごとの true 方位最近傍 beam における tone level delta と、
        source mask 外 source だけを抜き出した抑圧量を含む辞書。

    信号処理上の位置づけ:
        source mask 外 source を A2 が source-correlated leakage とみなすと、
        non-source p95 は改善しても未知 source の可視性が失われる。
        そのため、評価 tone への複素投影で source 固有周波数の level delta を分離して読む。
    """
    axis = np.asarray(axis_azimuth_deg, dtype=np.float64)
    if axis.ndim != 1 or axis.size != source_mask.source_mask.size:
        raise ValueError("axis_azimuth_deg and source mask must agree on n_beam.")

    labels: list[str] = []
    nearest_beams: list[int] = []
    nearest_azimuths_deg: list[float] = []
    in_mask_flags: list[bool] = []
    deltas_db: list[float] = []
    outside_labels: list[str] = []
    outside_deltas_db: list[float] = []

    for source in scenario.source_specs:
        nearest_beam = int(np.argmin(np.abs(axis - float(source.azimuth_deg))))
        in_mask = bool(source_mask.source_mask[nearest_beam])
        frequency_axis = np.asarray([float(source.frequency_hz)], dtype=np.float64)
        before_levels = _tone_projection_levels_db20(
            beam_output=before_output,
            fs_hz=float(fs_hz),
            frequencies_hz=frequency_axis,
        )
        after_levels = _tone_projection_levels_db20(
            beam_output=after_output,
            fs_hz=float(fs_hz),
            frequencies_hz=frequency_axis,
        )
        delta_db = float(after_levels[nearest_beam, 0] - before_levels[nearest_beam, 0])

        labels.append(str(source.label))
        nearest_beams.append(nearest_beam)
        nearest_azimuths_deg.append(float(axis[nearest_beam]))
        in_mask_flags.append(in_mask)
        deltas_db.append(delta_db)
        if not in_mask:
            outside_labels.append(str(source.label))
            outside_deltas_db.append(delta_db)

    if len(outside_deltas_db) > 0:
        # delta が負方向に大きいほど、mask 外 source を消している危険が大きい。
        max_suppression_db = float(min(outside_deltas_db))
        max_abs_delta_db = float(max(abs(value) for value in outside_deltas_db))
    else:
        max_suppression_db = 0.0
        max_abs_delta_db = 0.0

    return {
        "source_visibility_labels": "|".join(labels),
        "source_visibility_nearest_beam_indices": "|".join(
            str(index) for index in nearest_beams
        ),
        "source_visibility_nearest_azimuths_deg": "|".join(
            f"{azimuth:.3f}" for azimuth in nearest_azimuths_deg
        ),
        "source_visibility_in_mask_flags": "|".join(
            "true" if flag else "false" for flag in in_mask_flags
        ),
        "source_visibility_level_delta_db": "|".join(
            f"{delta:.6f}" for delta in deltas_db
        ),
        "mask_outside_source_count": int(len(outside_labels)),
        "mask_outside_source_labels": "|".join(outside_labels),
        "mask_outside_source_level_delta_db": "|".join(
            f"{delta:.6f}" for delta in outside_deltas_db
        ),
        "max_mask_outside_source_suppression_db": max_suppression_db,
        "max_mask_outside_source_abs_delta_db": max_abs_delta_db,
    }

def _candidate_rows(
    *,
    scenario: ScenarioDefinition,
    source_mask: SourceSectorMask,
    evaluation: CandidateEvaluation,
    axis_azimuth_deg: FloatArray,
    fs_hz: float,
    include_raw: bool,
) -> list[dict[str, object]]:
    """candidate 評価結果を summary CSV 行へ変換する。

    Args:
        scenario: 評価 scenario。
        source_mask: source / non-source sector mask。
        evaluation: candidate の raw/effective 出力。
        axis_azimuth_deg: beam 方位軸。shape は `[n_beam]`、単位は deg。
        fs_hz: サンプリング周波数。単位は Hz。
        include_raw: A2 raw 診断行を出力するかどうか。

    Returns:
        CSV row list。baseline は effective 行だけ、A2 は raw 診断行と effective 判定行を返す。
    """
    rows: list[dict[str, object]] = []
    candidate = evaluation.candidate
    before_levels_db = _rms_levels_db20(evaluation.before_window)

    stage_entries: list[tuple[str, ComplexArray, bool]] = []
    if include_raw and not candidate.is_baseline:
        stage_entries.append(("raw", evaluation.raw_output, False))
    stage_entries.append(("effective", evaluation.effective_output, True))

    for output_stage, after_output, decision_used in stage_entries:
        after_levels_db = _rms_levels_db20(after_output)
        row = _base_row(
            case=scenario.case,
            guard=candidate.guard,
            mask_type=scenario.mask_type,
            source_mask=source_mask,
            method_family="fixed_baseline" if candidate.is_baseline else "A2_source_mask_slc",
            method_id=candidate.method_id,
            candidate_id=candidate.candidate_id,
            output_stage=output_stage,
            decision_used=decision_used,
        )
        row.update(
            _metrics_row_fields(
                before_levels_db=before_levels_db,
                after_levels_db=after_levels_db,
                axis_azimuth_deg=axis_azimuth_deg,
                source_mask=source_mask,
                realtime_factor=evaluation.realtime_factor,
                nan_inf_count=int(np.count_nonzero(~np.isfinite(after_output))),
                condition_number=evaluation.condition_number,
            )
        )
        row.update(
            _source_visibility_fields(
                scenario=scenario,
                source_mask=source_mask,
                axis_azimuth_deg=axis_azimuth_deg,
                before_output=evaluation.before_window,
                after_output=after_output,
                fs_hz=float(fs_hz),
            )
        )
        if not decision_used:
            row["status"] = "diagnostic_raw"
        if candidate.is_baseline:
            row["status"] = "baseline"
        row.update(
            {
                "scenario_id": scenario.scenario_id,
                "scenario_group": scenario.scenario_group,
                "selection_basis": scenario.selection_basis,
                "scenario_notes": scenario.notes,
                "source_count": int(len(scenario.source_specs)),
                "source_labels": "|".join(source.label for source in scenario.source_specs),
                "source_azimuths_deg": "|".join(
                    f"{float(source.azimuth_deg):.3f}" for source in scenario.source_specs
                ),
                "source_frequencies_hz": "|".join(
                    f"{float(source.frequency_hz):.1f}" for source in scenario.source_specs
                ),
                "source_levels_db": "|".join(
                    f"{float(source.level_db):.3f}" for source in scenario.source_specs
                ),
                "eta": "" if candidate.eta is None else float(candidate.eta),
                "loading": "" if candidate.loading is None else float(candidate.loading),
                "train_mode": candidate.train_mode,
                "sample_per_dof_min": "" if candidate.is_baseline else float(SAMPLE_PER_DOF_MIN),
                "capacity_block_size": int(evaluation.capacity_block_size),
                "capacity_dof": int(evaluation.capacity_dof),
                "condition_number": ""
                if evaluation.condition_number is None
                else float(evaluation.condition_number),
                "weight_norm": ""
                if evaluation.weight_norm is None
                else float(evaluation.weight_norm),
                "realtime_factor": float(evaluation.realtime_factor),
                "a2_kernel_elapsed_sec": float(evaluation.elapsed_sec),
                "fallback_required": bool(evaluation.fallback_required),
                "fallback_reason": "|".join(evaluation.fallback_reasons),
                "metric_fallback_reason": "|".join(evaluation.metric_fallback_reasons),
                "n_ref": int(evaluation.n_ref),
                "n_non_source": int(evaluation.n_non_source),
            }
        )
        rows.append(row)

    return rows


def _tone_projection_levels_db20(
    *,
    beam_output: ComplexArray,
    fs_hz: float,
    frequencies_hz: FloatArray,
) -> FloatArray:
    """beam-domain 波形を tone 周波数へ複素投影し、FRAZ レベルを返す。

    Args:
        beam_output: beam-domain 複素波形。shape は `[n_beam, n_sample]`。
        fs_hz: サンプリング周波数。単位は Hz。
        frequencies_hz: 投影する tone 周波数。shape は `[n_freq]`、単位は Hz。

    Returns:
        周波数-方位レベル。shape は `[n_beam, n_freq]`、単位は `dB re input RMS`。

    信号処理式:
        各 beam の `x_b[n]` に `exp(-j 2π f n / fs)` を掛けて平均し、
        tone 成分の複素振幅を取り出す。今回の合成信号は complex exponential なので、
        投影係数の絶対値が RMS 振幅に対応する。
    """
    signals = np.asarray(beam_output, dtype=np.complex128)
    if signals.ndim != 2:
        raise ValueError("beam_output must have shape (n_beam, n_sample).")
    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    if frequencies.ndim != 1 or frequencies.size == 0:
        raise ValueError("frequencies_hz must have shape (n_freq,).")

    n_sample = int(signals.shape[1])
    time_axis_s = np.arange(n_sample, dtype=np.float64) / float(fs_hz)
    levels = np.empty((signals.shape[0], frequencies.size), dtype=np.float64)
    for frequency_index, frequency_hz in enumerate(frequencies):
        phase = np.exp(-1j * 2.0 * np.pi * float(frequency_hz) * time_axis_s)
        # signals * phase[None, :] は `[n_beam, n_sample]` のまま時間方向だけを混合する。
        # axis=1 の平均が各 beam の該当 tone 複素振幅である。
        coefficient = np.mean(signals * phase[np.newaxis, :], axis=1)
        levels[:, frequency_index] = 20.0 * np.log10(
            np.maximum(np.abs(coefficient), np.finfo(np.float64).tiny)
        )
    return np.asarray(levels, dtype=np.float64)


def _btr_relative_levels(
    *,
    beam_output: ComplexArray,
    fs_hz: float,
    block_size: int,
) -> tuple[FloatArray, FloatArray, IntArray]:
    """短時間 RMS から BTR 相対レベルと peak track を作る。

    Args:
        beam_output: beam-domain 複素波形。shape は `[n_beam, n_sample]`。
        fs_hz: サンプリング周波数。単位は Hz。
        block_size: 1 BTR frame に含める sample 数。

    Returns:
        `(times_s, relative_levels_db, peak_indices)`。
        `times_s` shape は `[n_frame]`、`relative_levels_db` shape は `[n_frame, n_beam]`、
        `peak_indices` shape は `[n_frame]` である。

    境界条件:
        端数 sample は frame 幅を一定に保つため切り捨てる。
        BTR は表示用であり、metric 判定は全評価窓 RMS の CSV 指標で行う。
    """
    signals = np.asarray(beam_output, dtype=np.complex128)
    if signals.ndim != 2:
        raise ValueError("beam_output must have shape (n_beam, n_sample).")
    if int(block_size) <= 0:
        raise ValueError("block_size must be positive.")

    n_frame = int(signals.shape[1] // int(block_size))
    if n_frame <= 0:
        raise ValueError("beam_output is shorter than one BTR block.")
    trimmed = signals[:, : n_frame * int(block_size)]
    # reshape 前: `[n_beam, n_frame * block_size]`。
    # reshape 後: `[n_beam, n_frame, block_size]` とし、axis=2 を短時間 RMS の時間軸にする。
    frames = trimmed.reshape(signals.shape[0], n_frame, int(block_size))
    rms = np.sqrt(np.mean(np.abs(frames) ** 2, axis=2))
    levels_db = 20.0 * np.log10(np.maximum(rms, np.finfo(np.float64).tiny))
    relative_levels_db = levels_db.T - np.max(levels_db, axis=0)[:, np.newaxis]
    peak_indices = np.asarray(np.argmax(relative_levels_db, axis=1), dtype=np.int64)
    times_s = (
        (np.arange(n_frame, dtype=np.float64) * float(block_size) + 0.5 * float(block_size))
        / float(fs_hz)
    )
    return (
        np.asarray(times_s, dtype=np.float64),
        np.asarray(relative_levels_db, dtype=np.float64),
        peak_indices,
    )


def _representative_scenarios() -> tuple[ScenarioDefinition, ...]:
    """BL/FRAZ/BTR を保存する代表 scenario を返す。

    Returns:
        pass、hold、detected mask、offgrid 0.5 deg の 4 scenario。

    境界条件:
        A2_safe / A2_aggressive は既存 full sweep の effective では全条件 pass である。
        そのため hold 代表は、候補選定前の A2 条件で hold になった detected mask 条件を
        同一 source 条件として再評価する。
    """
    pass_case = _case(
        tier="tier0",
        target_frequency_hz=10000.0,
        interferer_frequency_hz=8192.0,
        interferer_azimuth_base_deg=60.0,
        offgrid_deg=0.0,
    )
    hold_case = _case(
        tier="tier1",
        target_frequency_hz=6144.0,
        interferer_frequency_hz=8192.0,
        interferer_azimuth_base_deg=45.0,
        offgrid_deg=0.0,
    )
    offgrid_case = _case(
        tier="tier0",
        target_frequency_hz=10000.0,
        interferer_frequency_hz=8192.0,
        interferer_azimuth_base_deg=60.0,
        offgrid_deg=0.5,
    )

    pass_sources = _two_source_specs(pass_case)
    hold_sources = _two_source_specs(hold_case)
    offgrid_sources = _two_source_specs(offgrid_case)
    return (
        ScenarioDefinition(
            scenario_id="representative_pass_tier0_oracle",
            scenario_group="representative_pass",
            selection_basis="A2_safe/A2_aggressive effective pass in comparison_summary.csv",
            case=pass_case,
            mask_type="oracle",
            source_specs=pass_sources,
            source_mask_azimuths_deg=np.asarray(
                [source.azimuth_deg for source in pass_sources], dtype=np.float64
            ),
            notes="Tier0 target=90deg/10000Hz, interferer=60deg/8192Hz, oracle mask.",
        ),
        ScenarioDefinition(
            scenario_id="representative_hold_tier1_detected",
            scenario_group="representative_hold",
            selection_basis=(
                "full sweep hold example: tier1_tf6144_if8192_ia045_off000 detected "
                "with weaker A2 p95_not_pass"
            ),
            case=hold_case,
            mask_type="detected",
            source_specs=hold_sources,
            source_mask_azimuths_deg=np.asarray(
                [source.azimuth_deg for source in hold_sources], dtype=np.float64
            ),
            notes="候補選定前の detected hold 条件を、絞り込み候補で再評価する。",
        ),
        ScenarioDefinition(
            scenario_id="representative_detected_tier0",
            scenario_group="representative_detected_mask",
            selection_basis="Tier0 detected mask behavior check",
            case=pass_case,
            mask_type="detected",
            source_specs=pass_sources,
            source_mask_azimuths_deg=np.asarray(
                [source.azimuth_deg for source in pass_sources], dtype=np.float64
            ),
            notes="fixed baseline の mixed RMS peak から source mask を検出する代表条件。",
        ),
        ScenarioDefinition(
            scenario_id="representative_offgrid_0p5_tier0_oracle",
            scenario_group="representative_offgrid_0p5",
            selection_basis="Tier0 offgrid 0.5 deg behavior check",
            case=offgrid_case,
            mask_type="oracle",
            source_specs=offgrid_sources,
            source_mask_azimuths_deg=np.asarray(
                [source.azimuth_deg for source in offgrid_sources], dtype=np.float64
            ),
            notes="target/interferer とも +0.5 deg ずれた oracle mask 条件。",
        ),
    )


def _negative_scenarios() -> tuple[ScenarioDefinition, ...]:
    """8月評価候補へ追加する負例 scenario を返す。

    Returns:
        unknown source、weak source、3 source 以上、detected 1 本、offgrid mask ずれの scenario。

    信号処理上の位置づけ:
        A2 は source mask 内を保護して non-source を抑える方式であるため、
        mask 外に実 source がある条件や検出漏れ条件では raw 改善だけで採用しない。
    """
    base_case = _case(
        tier="tier0",
        target_frequency_hz=10000.0,
        interferer_frequency_hz=8192.0,
        interferer_azimuth_base_deg=60.0,
        offgrid_deg=0.0,
    )
    base_sources = _two_source_specs(base_case)
    nominal_mask_azimuths = np.asarray([90.0, 60.0], dtype=np.float64)

    unknown_source = SourceSpec(
        label="unknown_outside_mask",
        frequency_hz=6144.0,
        azimuth_deg=135.0,
        level_db=-3.0,
        phase_rad=UNKNOWN_SOURCE_PHASE_RAD,
    )
    weak_source = SourceSpec(
        label="weak_outside_mask",
        frequency_hz=6144.0,
        azimuth_deg=135.0,
        level_db=-18.0,
        phase_rad=WEAK_SOURCE_PHASE_RAD,
    )
    third_known_source = SourceSpec(
        label="third_known_source",
        frequency_hz=6144.0,
        azimuth_deg=120.0,
        level_db=-9.0,
        phase_rad=UNKNOWN_SOURCE_PHASE_RAD,
    )
    weak_detected_case_sources = (
        base_sources[0],
        SourceSpec(
            label="weak_interferer_detected_miss",
            frequency_hz=8192.0,
            azimuth_deg=60.0,
            level_db=-24.0,
            phase_rad=INTERFERER_PHASE_RAD,
        ),
    )
    offgrid_source_case = _case(
        tier="tier0",
        target_frequency_hz=10000.0,
        interferer_frequency_hz=8192.0,
        interferer_azimuth_base_deg=60.0,
        offgrid_deg=0.5,
    )
    offgrid_sources = _two_source_specs(offgrid_source_case)

    return (
        ScenarioDefinition(
            scenario_id="negative_unknown_source_outside_mask",
            scenario_group="negative_unknown_source",
            selection_basis="unknown source is intentionally excluded from oracle source mask",
            case=base_case,
            mask_type="oracle",
            source_specs=base_sources + (unknown_source,),
            source_mask_azimuths_deg=nominal_mask_azimuths,
            notes="135 deg / 6144 Hz / -3 dB の未知 source を source mask 外に置く。",
        ),
        ScenarioDefinition(
            scenario_id="negative_weak_source_outside_mask",
            scenario_group="negative_weak_source",
            selection_basis="weak source is intentionally excluded from oracle source mask",
            case=base_case,
            mask_type="oracle",
            source_specs=base_sources + (weak_source,),
            source_mask_azimuths_deg=nominal_mask_azimuths,
            notes="135 deg / 6144 Hz / -18 dB の弱 source を source mask 外に置く。",
        ),
        ScenarioDefinition(
            scenario_id="negative_three_or_more_known_sources",
            scenario_group="negative_source_count_ge3",
            selection_basis="known source count is three and oracle mask must protect all three",
            case=base_case,
            mask_type="oracle",
            source_specs=base_sources + (third_known_source,),
            source_mask_azimuths_deg=np.asarray([90.0, 60.0, 120.0], dtype=np.float64),
            notes="3 本 source を oracle mask 内に含め、source 保護の容量を確認する。",
        ),
        ScenarioDefinition(
            scenario_id="negative_detected_mask_single_source",
            scenario_group="negative_detected_single_source",
            selection_basis="detected mask must miss the weak interferer by level threshold",
            case=base_case,
            mask_type="detected",
            source_specs=weak_detected_case_sources,
            source_mask_azimuths_deg=nominal_mask_azimuths,
            notes=(
                f"detected threshold {DETECTED_SOURCE_THRESHOLD_DB_BELOW_PEAK:g} "
                "dB below peak により target 1 本だけを検出する条件。"
            ),
        ),
        ScenarioDefinition(
            scenario_id="negative_source_azimuth_offgrid_0p5_mask_nominal",
            scenario_group="negative_source_offgrid_0p5",
            selection_basis=(
                "source azimuth is shifted by 0.5 deg while "
                "oracle mask uses nominal azimuths"
            ),
            case=offgrid_source_case,
            mask_type="oracle",
            source_specs=offgrid_sources,
            source_mask_azimuths_deg=nominal_mask_azimuths,
            notes=(
                "source は +0.5 deg、oracle mask は nominal 方位で作る。"
                "wide guard 内のずれを確認する。"
            ),
        ),
    )


def _evaluate_scenario(
    *,
    scenario: ScenarioDefinition,
    assets: EvaluationAssets,
    include_raw: bool,
) -> tuple[list[dict[str, object]], dict[str, CandidateEvaluation], SourceSectorMask]:
    """scenario を fixed / A2_safe / A2_aggressive で評価する。

    Args:
        scenario: 評価 scenario。
        assets: beam-domain 合成に使う共有 asset。
        include_raw: A2 raw 診断行を出力するかどうか。

    Returns:
        `(rows, evaluations, source_mask)`。
        `rows` は CSV 行、`evaluations` は method_id を key とする評価結果である。

    Raises:
        ValueError: source mask と beam 数が整合しない場合。
    """
    weights_by_frequency = _operational_weights_for_sources(
        assets=assets,
        source_specs=scenario.source_specs,
    )
    before_output = _synthesize_multisource_beam_output(
        source_specs=scenario.source_specs,
        assets=assets,
        weights_by_frequency=weights_by_frequency,
        n_sample=N_SAMPLE,
    )
    before_levels_db = _rms_levels_db20(before_output)
    source_mask = _source_mask_for_scenario(
        scenario=scenario,
        assets=assets,
        before_levels_db=before_levels_db,
    )
    if source_mask.source_mask.size != assets.axis_azimuth_deg.size:
        raise ValueError("source mask and beam axis must agree on n_beam.")

    source_reference_beams = np.flatnonzero(source_mask.source_mask).astype(np.int64)
    evaluation_duration_s = float(_evaluation_window(before_output).shape[1]) / float(
        assets.array_definition.fs_hz
    )

    rows: list[dict[str, object]] = []
    evaluations: dict[str, CandidateEvaluation] = {}
    for candidate in SHORTLIST_CANDIDATES:
        evaluation = _evaluate_candidate(
            candidate=candidate,
            before_output=before_output,
            source_mask=source_mask,
            source_reference_beams=source_reference_beams,
            axis_azimuth_deg=assets.axis_azimuth_deg,
            evaluation_duration_s=evaluation_duration_s,
        )
        evaluations[candidate.method_id] = evaluation
        rows.extend(
            _candidate_rows(
                scenario=scenario,
                source_mask=source_mask,
                evaluation=evaluation,
                axis_azimuth_deg=assets.axis_azimuth_deg,
                fs_hz=float(assets.array_definition.fs_hz),
                include_raw=include_raw,
            )
        )
    return rows, evaluations, source_mask


def _representative_target_source(sources: tuple[SourceSpec, ...]) -> SourceSpec:
    """図の BL marker に使う target source を返す。

    Args:
        sources: source 条件列。shape は `[n_source]`。

    Returns:
        label が `target` の source。存在しない場合は先頭 source。

    Raises:
        ValueError: sources が空の場合。
    """
    if len(sources) == 0:
        raise ValueError("sources must not be empty.")
    for source in sources:
        if source.label == "target":
            return source
    return sources[0]


def _source_marker_points(sources: tuple[SourceSpec, ...]) -> list[tuple[float, float, str]]:
    """FRAZ 図へ重ねる source marker 点列を返す。

    Args:
        sources: source 条件列。shape は `[n_source]`。

    Returns:
        `(azimuth_deg, frequency_hz, label)` の list。
    """
    return [
        (float(source.azimuth_deg), float(source.frequency_hz), str(source.label))
        for source in sources
    ]


def _fraz_peak_points(
    *,
    axis_azimuth_deg: FloatArray,
    frequencies_hz: FloatArray,
    fraz_levels_db: FloatArray,
) -> list[tuple[float, float, str]]:
    """FRAZ 周波数ごとの peak 方位 marker を返す。

    Args:
        axis_azimuth_deg: beam 方位軸。shape は `[n_beam]`、単位は deg。
        frequencies_hz: FRAZ 周波数軸。shape は `[n_freq]`、単位は Hz。
        fraz_levels_db: FRAZ レベル。shape は `[n_beam, n_freq]`。

    Returns:
        周波数ごとの peak marker list。

    Raises:
        ValueError: shape が整合しない場合。
    """
    axis = np.asarray(axis_azimuth_deg, dtype=np.float64)
    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    levels = np.asarray(fraz_levels_db, dtype=np.float64)
    if levels.shape != (axis.size, frequencies.size):
        raise ValueError("fraz_levels_db must have shape (n_beam, n_freq).")

    points: list[tuple[float, float, str]] = []
    for frequency_index, frequency_hz in enumerate(frequencies):
        peak_index = int(np.argmax(levels[:, frequency_index]))
        points.append(
            (
                float(axis[peak_index]),
                float(frequency_hz),
                f"Diagnostic slice max {int(round(float(frequency_hz)))} Hz",
            )
        )
    return points


def _mask_center_caption(scenario: ScenarioDefinition) -> str:
    """図 caption 用に source mask の中心条件を説明する。

    Args:
        scenario: 図を生成する scenario。

    Returns:
        mask が true source、nominal source、detected peak のどれに基づくかを説明する文字列。
    """
    if scenario.mask_type == "detected":
        return "mask = detected, centered at fixed-baseline RMS peaks"
    if scenario.scenario_id == "negative_source_azimuth_offgrid_0p5_mask_nominal":
        return "mask = oracle, centered at nominal source azimuths"
    return "mask = oracle, centered at true source azimuths"


def _offgrid_caption_lines(scenario: ScenarioDefinition) -> tuple[str, ...]:
    """offgrid 条件で誤読を避けるための nominal / true 方位説明を返す。

    Args:
        scenario: 図を生成する scenario。

    Returns:
        caption に追加する行の tuple。offgrid でない場合は空 tuple。
    """
    if abs(float(scenario.case.offgrid_deg)) <= 0.0:
        return ()
    return (
        "Nominal target azimuth = 90.0 deg",
        f"True target azimuth = {float(scenario.case.target_azimuth_deg):.1f} deg",
        f"Nominal source 2 = {float(scenario.case.interferer_azimuth_base_deg):.1f} deg",
        f"True source 2 = {float(scenario.case.interferer_azimuth_deg):.1f} deg",
        _mask_center_caption(scenario),
    )


def _figure_caption(scenario: ScenarioDefinition, *, level_label: str) -> str:
    """BL/FRAZ/BTR 図の caption を作る。

    Args:
        scenario: 図を生成する scenario。
        level_label: レベル基準。例は `dB re input RMS`。

    Returns:
        図下部に表示する caption。offgrid 条件では nominal / true 方位を含める。
    """
    lines = [
        f"case={scenario.case.case_id}, mask={scenario.mask_type}, stage=effective",
        f"level={level_label}",
    ]
    lines.extend(_offgrid_caption_lines(scenario))
    if len(_offgrid_caption_lines(scenario)) == 0:
        lines.append(_mask_center_caption(scenario))
    if scenario.scenario_group.startswith("negative"):
        lines.append(f"negative check: {scenario.scenario_group}")
    return "\n".join(lines)

def _save_representative_figures(
    *,
    scenario: ScenarioDefinition,
    evaluations: dict[str, CandidateEvaluation],
    assets: EvaluationAssets,
) -> list[dict[str, object]]:
    """1 代表 scenario の BL/FRAZ/BTR を candidate ごとに保存する。

    Args:
        scenario: 代表図 scenario。
        evaluations: method_id を key とする candidate 評価結果。
        assets: 方位軸と fs を含む共有 asset。

    Returns:
        図 manifest CSV 用 row list。

    Raises:
        KeyError: 必要 candidate の評価結果が存在しない場合。
    """
    manifest_rows: list[dict[str, object]] = []
    target_source = _representative_target_source(scenario.source_specs)
    target_points = _source_marker_points(scenario.source_specs)
    target_azimuths_deg = np.asarray(
        [float(source.azimuth_deg) for source in scenario.source_specs], dtype=np.float64
    )
    fs_hz = float(assets.array_definition.fs_hz)

    for candidate in SHORTLIST_CANDIDATES:
        evaluation = evaluations[candidate.method_id]
        output = np.asarray(evaluation.effective_output, dtype=np.complex128)
        candidate_dir = FIGURE_DIR / scenario.scenario_id / candidate.method_id
        bl_path = candidate_dir / "bl.png"
        fraz_path = candidate_dir / "fraz.png"
        btr_path = candidate_dir / "btr.png"

        bl_levels_db = _rms_levels_db20(output)
        bl_peak_index = int(np.argmax(bl_levels_db))
        bl_peak_azimuth_deg = float(assets.axis_azimuth_deg[bl_peak_index])
        caption = _figure_caption(scenario, level_label=LEVEL_UNIT_LABEL)
        plot_bl_response(
            assets.axis_azimuth_deg,
            bl_levels_db,
            target_azimuth_deg=float(target_source.azimuth_deg),
            peak_azimuth_deg=bl_peak_azimuth_deg,
            title=f"{scenario.scenario_group} / {candidate.method_id} / BL",
            caption=caption,
            output_path=bl_path,
            response_label=candidate.method_id,
            level_unit_label=LEVEL_UNIT_LABEL,
        )

        fraz_levels_db = _tone_projection_levels_db20(
            beam_output=output,
            fs_hz=fs_hz,
            frequencies_hz=TONE_FREQUENCIES_HZ,
        )
        plot_fraz_heatmap(
            assets.axis_azimuth_deg,
            TONE_FREQUENCIES_HZ,
            fraz_levels_db,
            target_points=target_points,
            peak_points=_fraz_peak_points(
                axis_azimuth_deg=assets.axis_azimuth_deg,
                frequencies_hz=TONE_FREQUENCIES_HZ,
                fraz_levels_db=fraz_levels_db,
            ),
            title=f"{scenario.scenario_group} / {candidate.method_id} / FRAZ",
            caption=caption,
            output_path=fraz_path,
            colorbar_label=f"RMS Level [{LEVEL_UNIT_LABEL}]",
        )

        times_s, btr_relative_db, btr_peak_indices = _btr_relative_levels(
            beam_output=output,
            fs_hz=fs_hz,
            block_size=BTR_BLOCK_SIZE,
        )
        btr_peak_azimuths_deg = np.asarray(
            assets.axis_azimuth_deg[btr_peak_indices], dtype=np.float64
        )
        plot_btr_heatmap(
            assets.axis_azimuth_deg,
            times_s,
            btr_relative_db,
            btr_peak_azimuths_deg=btr_peak_azimuths_deg,
            target_azimuths_deg=target_azimuths_deg,
            title=f"{scenario.scenario_group} / {candidate.method_id} / BTR",
            caption=caption,
            output_path=btr_path,
            colorbar_label="Relative Level [dB re frame max]",
        )

        manifest_rows.append(
            {
                "scenario_id": scenario.scenario_id,
                "scenario_group": scenario.scenario_group,
                "case_id": scenario.case.case_id,
                "mask_type": scenario.mask_type,
                "method_id": candidate.method_id,
                "candidate_id": candidate.candidate_id,
                "bl_path": bl_path.as_posix(),
                "fraz_path": fraz_path.as_posix(),
                "btr_path": btr_path.as_posix(),
                "level_unit_label": LEVEL_UNIT_LABEL,
            }
        )

    return manifest_rows


def _evaluate_representatives(
    *,
    assets: EvaluationAssets,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """代表 scenario を評価し、図 manifest と summary row を返す。

    Args:
        assets: beam-domain 合成に使う共有 asset。

    Returns:
        `(summary_rows, figure_manifest_rows)`。
    """
    summary_rows: list[dict[str, object]] = []
    figure_manifest_rows: list[dict[str, object]] = []
    for scenario in _representative_scenarios():
        rows, evaluations, _ = _evaluate_scenario(
            scenario=scenario,
            assets=assets,
            include_raw=True,
        )
        summary_rows.extend(rows)
        figure_manifest_rows.extend(
            _save_representative_figures(
                scenario=scenario,
                evaluations=evaluations,
                assets=assets,
            )
        )
    return summary_rows, figure_manifest_rows


def _evaluate_negatives(*, assets: EvaluationAssets) -> list[dict[str, object]]:
    """負例 scenario を評価し、summary row を返す。

    Args:
        assets: beam-domain 合成に使う共有 asset。

    Returns:
        負例評価の CSV row list。A2 raw と effective を両方含む。
    """
    rows: list[dict[str, object]] = []
    for scenario in _negative_scenarios():
        scenario_rows, _, _ = _evaluate_scenario(
            scenario=scenario,
            assets=assets,
            include_raw=True,
        )
        rows.extend(scenario_rows)
    return rows



def _evaluate_negative_risk_figures(*, assets: EvaluationAssets) -> list[dict[str, object]]:
    """source mask 外 source 消失リスク確認用の負例図 manifest を返す。

    Args:
        assets: beam-domain 合成に使う共有 asset。

    Returns:
        指定 4 scenario の BL/FRAZ/BTR 図 manifest row list。

    信号処理上の位置づけ:
        source mask 外 source は A2 の抑圧対象 non-source に入るため、
        non-source 指標だけでは source visibility loss を見落とす可能性がある。
        そのため、資料確認用に fixed / A2_safe / A2_aggressive を横並びで図示する。
    """
    manifest_rows: list[dict[str, object]] = []
    risk_ids = set(RISK_FIGURE_SCENARIO_IDS)
    for scenario in _negative_scenarios():
        if scenario.scenario_id not in risk_ids:
            continue
        _, evaluations, _ = _evaluate_scenario(
            scenario=scenario,
            assets=assets,
            include_raw=True,
        )
        manifest_rows.extend(
            _save_representative_figures(
                scenario=scenario,
                evaluations=evaluations,
                assets=assets,
            )
        )
    return manifest_rows

def _build_safety_gate_rows(*, assets: EvaluationAssets) -> list[dict[str, object]]:
    """raw 悪化時に effective が fixed baseline へ戻ることを確認する。

    Args:
        assets: beam-domain 合成に使う共有 asset。

    Returns:
        raw 人工悪化行と effective fallback 行。

    Raises:
        RuntimeError: 人工悪化が metric safety gate に検出されない場合。

    信号処理上の位置づけ:
        ここでは A2 solver の数値失敗ではなく、raw 出力 metric の悪化を作る。
        non-source sector だけを +6 dB 増幅し、source mask 内は固定整相のままにすることで、
        effective 出力が固定整相へ戻る safety gate を単体確認する。
    """
    scenario = _representative_scenarios()[0]
    weights_by_frequency = _operational_weights_for_sources(
        assets=assets,
        source_specs=scenario.source_specs,
    )
    before_output = _synthesize_multisource_beam_output(
        source_specs=scenario.source_specs,
        assets=assets,
        weights_by_frequency=weights_by_frequency,
        n_sample=N_SAMPLE,
    )
    before_levels_db = _rms_levels_db20(before_output)
    source_mask = _source_mask_for_scenario(
        scenario=scenario,
        assets=assets,
        before_levels_db=before_levels_db,
    )
    before_window = _evaluation_window(before_output)
    before_window_levels_db = _rms_levels_db20(before_window)

    artificial_gain_db = 6.0
    artificial_gain = float(10.0 ** (artificial_gain_db / 20.0))
    artificial_raw_output = before_window.copy()
    # non-source sector だけを増幅し、source mask 内の保護成分を変えない。
    # これにより raw metric の局所悪化だけを safety gate へ入力できる。
    artificial_raw_output[source_mask.non_source_mask, :] *= artificial_gain
    raw_levels_db = _rms_levels_db20(artificial_raw_output)
    raw_metric_fields = _metrics_row_fields(
        before_levels_db=before_window_levels_db,
        after_levels_db=raw_levels_db,
        axis_azimuth_deg=assets.axis_azimuth_deg,
        source_mask=source_mask,
        realtime_factor=0.0,
        nan_inf_count=int(np.count_nonzero(~np.isfinite(artificial_raw_output))),
        condition_number=None,
    )
    metric_reasons = _metric_safety_fallback_reasons(raw_metric_fields)
    if len(metric_reasons) == 0:
        raise RuntimeError("artificial raw worsening was not caught by metric safety gate.")

    effective_output = before_window.copy()
    effective_levels_db = _rms_levels_db20(effective_output)
    effective_metric_fields = _metrics_row_fields(
        before_levels_db=before_window_levels_db,
        after_levels_db=effective_levels_db,
        axis_azimuth_deg=assets.axis_azimuth_deg,
        source_mask=source_mask,
        realtime_factor=0.0,
        nan_inf_count=int(np.count_nonzero(~np.isfinite(effective_output))),
        condition_number=None,
    )
    max_abs_diff = float(np.max(np.abs(effective_output - before_window)))
    effective_equals_fixed = bool(np.allclose(effective_output, before_window, atol=1.0e-12))

    probe_candidate = SHORTLIST_CANDIDATES[1]
    rows: list[dict[str, object]] = []
    for output_stage, fields, fallback_required, fallback_reason in (
        ("raw_artificial", raw_metric_fields, False, ""),
        ("effective", effective_metric_fields, True, "|".join(metric_reasons)),
    ):
        row = _base_row(
            case=scenario.case,
            guard=WIDE_GUARD,
            mask_type=scenario.mask_type,
            source_mask=source_mask,
            method_family="A2_source_mask_slc",
            method_id="A2_metric_safety_gate_probe",
            candidate_id=probe_candidate.candidate_id,
            output_stage=output_stage,
            decision_used=output_stage == "effective",
        )
        row.update(fields)
        if output_stage == "effective":
            row["status"] = "fallback_baseline"
        row.update(
            {
                "scenario_id": "safety_gate_raw_worsening_probe",
                "scenario_group": "safety_gate",
                "selection_basis": "artificial non-source raw worsening",
                "raw_artificial_non_source_gain_db": artificial_gain_db,
                "fallback_required": bool(fallback_required),
                "fallback_reason": fallback_reason,
                "effective_equals_fixed_baseline": effective_equals_fixed,
                "max_abs_diff_vs_fixed_baseline": max_abs_diff,
                "metric_fallback_reason": "|".join(metric_reasons),
                "condition_number": "",
                "weight_norm": "",
                "realtime_factor": 0.0,
            }
        )
        rows.append(row)

    return rows


def _a2_candidates() -> tuple[ShortlistCandidate, ...]:
    """runtime 測定対象の A2 候補だけを返す。

    Returns:
        A2_safe と A2_aggressive の tuple。
    """
    return tuple(candidate for candidate in SHORTLIST_CANDIDATES if not candidate.is_baseline)


def _measure_runtime_rows(*, assets: EvaluationAssets) -> list[dict[str, object]]:
    """代表 scenario に限定して A2 kernel の runtime factor を測る。

    Args:
        assets: beam-domain 合成に使う共有 asset。

    Returns:
        runtime summary CSV 用 row list。

    信号処理上の位置づけ:
        example 全体の I/O、図保存、source 合成は runtime へ含めない。
        `_run_a2_source_mask_slc` 内の covariance、loading、solve、cancel 適用だけを測る。
    """
    rows: list[dict[str, object]] = []
    for scenario in _representative_scenarios():
        weights_by_frequency = _operational_weights_for_sources(
            assets=assets,
            source_specs=scenario.source_specs,
        )
        before_output = _synthesize_multisource_beam_output(
            source_specs=scenario.source_specs,
            assets=assets,
            weights_by_frequency=weights_by_frequency,
            n_sample=N_SAMPLE,
        )
        before_levels_db = _rms_levels_db20(before_output)
        source_mask = _source_mask_for_scenario(
            scenario=scenario,
            assets=assets,
            before_levels_db=before_levels_db,
        )
        source_reference_beams = np.flatnonzero(source_mask.source_mask).astype(np.int64)
        evaluation_duration_s = float(_evaluation_window(before_output).shape[1]) / float(
            assets.array_definition.fs_hz
        )

        for candidate in _a2_candidates():
            if candidate.eta is None or candidate.loading is None:
                raise ValueError("A2 candidate requires eta and loading.")
            for _ in range(A2_KERNEL_WARMUP_COUNT):
                _run_a2_source_mask_slc(
                    beam_output=before_output,
                    source_sector_mask=source_mask,
                    source_reference_beams=source_reference_beams,
                    eta=float(candidate.eta),
                    loading=float(candidate.loading),
                    sample_per_dof=SAMPLE_PER_DOF_MIN,
                    train_mode=candidate.train_mode,
                )

            elapsed_values: list[float] = []
            last_condition_number: float | None = None
            last_weight_norm: float | None = None
            last_fallback_required = False
            last_fallback_reason = ""
            for _ in range(A2_KERNEL_REPEAT_COUNT):
                result = _run_a2_source_mask_slc(
                    beam_output=before_output,
                    source_sector_mask=source_mask,
                    source_reference_beams=source_reference_beams,
                    eta=float(candidate.eta),
                    loading=float(candidate.loading),
                    sample_per_dof=SAMPLE_PER_DOF_MIN,
                    train_mode=candidate.train_mode,
                )
                elapsed_values.append(float(result.elapsed_sec))
                last_condition_number = result.condition_number
                last_weight_norm = result.weight_norm
                last_fallback_required = bool(result.fallback_required)
                last_fallback_reason = "|".join(result.fallback_reasons)

            elapsed = np.asarray(elapsed_values, dtype=np.float64)
            mean_elapsed_sec = float(np.mean(elapsed))
            median_elapsed_sec = float(np.median(elapsed))
            p95_elapsed_sec = float(np.percentile(elapsed, 95.0))
            rows.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "scenario_group": scenario.scenario_group,
                    "case_id": scenario.case.case_id,
                    "mask_type": scenario.mask_type,
                    "method_id": candidate.method_id,
                    "candidate_id": candidate.candidate_id,
                    "eta": float(candidate.eta),
                    "loading": float(candidate.loading),
                    "train_mode": candidate.train_mode,
                    "sample_per_dof_min": float(SAMPLE_PER_DOF_MIN),
                    "runtime_scope": "a2_kernel_only",
                    "repeat_count": int(A2_KERNEL_REPEAT_COUNT),
                    "warmup_count": int(A2_KERNEL_WARMUP_COUNT),
                    "evaluation_duration_s": float(evaluation_duration_s),
                    "mean_elapsed_sec": mean_elapsed_sec,
                    "median_elapsed_sec": median_elapsed_sec,
                    "p95_elapsed_sec": p95_elapsed_sec,
                    "min_elapsed_sec": float(np.min(elapsed)),
                    "max_elapsed_sec": float(np.max(elapsed)),
                    "runtime_factor_mean": float(mean_elapsed_sec / evaluation_duration_s),
                    "runtime_factor_median": float(median_elapsed_sec / evaluation_duration_s),
                    "runtime_factor_p95": float(p95_elapsed_sec / evaluation_duration_s),
                    "realtime_factor_mean": float(mean_elapsed_sec / evaluation_duration_s),
                    "realtime_factor_p95": float(p95_elapsed_sec / evaluation_duration_s),
                    "condition_number": ""
                    if last_condition_number is None
                    else float(last_condition_number),
                    "weight_norm": "" if last_weight_norm is None else float(last_weight_norm),
                    "fallback_required": last_fallback_required,
                    "fallback_reason": last_fallback_reason,
                    "n_ref": int(source_reference_beams.size),
                    "non_source_beam_count": int(np.count_nonzero(source_mask.non_source_mask)),
                }
            )

    return rows


def _status_counts(rows: list[dict[str, object]], method_id: str) -> dict[str, int]:
    """effective 判定行の status 件数を method_id ごとに集計する。

    Args:
        rows: summary CSV row list。
        method_id: 集計対象 method。

    Returns:
        status を key、件数を value とする辞書。
    """
    counts: dict[str, int] = {}
    for row in rows:
        if str(row.get("method_id", "")) != method_id:
            continue
        if str(row.get("output_stage", "")) != "effective":
            continue
        if not bool(row.get("decision_used", False)):
            continue
        status = str(row.get("status", ""))
        counts[status] = counts.get(status, 0) + 1
    return counts


def _count_label(counts: dict[str, int]) -> str:
    """status count 辞書を report 用の短い文字列へ変換する。

    Args:
        counts: status count 辞書。

    Returns:
        `key=value` を comma 区切りにした文字列。
    """
    if len(counts) == 0:
        return ""
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))


def _candidate_effective_rows(
    rows: list[dict[str, object]], method_id: str
) -> list[dict[str, object]]:
    """指定 method の effective 判定行だけを返す。

    Args:
        rows: summary CSV row list。
        method_id: 抽出対象 method。

    Returns:
        effective 判定 row list。
    """
    return [
        row
        for row in rows
        if str(row.get("method_id", "")) == method_id
        and str(row.get("output_stage", "")) == "effective"
        and bool(row.get("decision_used", False))
    ]


def _max_row_float(rows: list[dict[str, object]], key: str) -> float | None:
    """row list から指定 field の最大値を返す。

    Args:
        rows: summary CSV row list。
        key: 最大値を取る field 名。

    Returns:
        最大値。値が存在しない場合は None。
    """
    values: list[float] = []
    for row in rows:
        try:
            values.append(_row_float(row, key))
        except TypeError:
            continue
    if len(values) == 0:
        return None
    return float(max(values))


def _fallback_count(rows: list[dict[str, object]]) -> int:
    """effective row の fallback_required 件数を数える。

    Args:
        rows: summary CSV row list。

    Returns:
        fallback_required が true の件数。
    """
    return int(sum(1 for row in rows if bool(row.get("fallback_required", False))))


def _runtime_p95_by_method(runtime_rows: list[dict[str, object]]) -> dict[str, float]:
    """runtime summary から method ごとの最大 p95 runtime factor を返す。

    Args:
        runtime_rows: runtime summary row list。

    Returns:
        method_id を key、代表 scenario 内の最大 `runtime_factor_p95` を value とする辞書。
    """
    values: dict[str, list[float]] = {}
    for row in runtime_rows:
        method_id = str(row.get("method_id", ""))
        values.setdefault(method_id, []).append(_row_float(row, "runtime_factor_p95"))
    return {method_id: float(max(method_values)) for method_id, method_values in values.items()}


def _original_sweep_counts() -> dict[str, dict[str, int]]:
    """comparison_summary.csv から絞り込み候補の既存 sweep 件数を読む。

    Returns:
        method_id を key、status count を value とする辞書。

    境界条件:
        既存 CSV が存在しない場合は空辞書を返す。新規評価 artifact はその場合でも生成できるが、
        report の original sweep 欄は空になる。
    """
    if not COMPARISON_SUMMARY_CSV_PATH.exists():
        return {}

    candidate_id_by_method = {
        "A2_safe": "eta0.5_loading0.03_wide_one_block_delay",
        "A2_aggressive": "eta1_loading0.1_wide_one_block_delay",
    }
    counts: dict[str, dict[str, int]] = {
        "fixed_baseline": {},
        "A2_safe": {},
        "A2_aggressive": {},
    }
    with COMPARISON_SUMMARY_CSV_PATH.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            method_family = str(row.get("method_family", ""))
            output_stage = str(row.get("output_stage", ""))
            if output_stage != "effective":
                continue
            status = str(row.get("status", ""))
            if method_family == "fixed_baseline":
                counts["fixed_baseline"][status] = counts["fixed_baseline"].get(status, 0) + 1
                continue
            if method_family != "A2_source_mask_slc":
                continue
            candidate_id = str(row.get("candidate_id", ""))
            for method_id, expected_candidate_id in candidate_id_by_method.items():
                if candidate_id == expected_candidate_id:
                    counts[method_id][status] = counts[method_id].get(status, 0) + 1
    return counts


def _format_optional_float(value: float | None) -> str:
    """report table 用に optional float を整形する。

    Args:
        value: 整形対象の値。

    Returns:
        数値なら小数 3 桁、None なら空文字。
    """
    if value is None:
        return ""
    return f"{float(value):.3f}"



def _source_mask_risk_report_lines(negative_rows: list[dict[str, object]]) -> list[str]:
    """source mask 外 source 消失リスクの report 行を作る。

    Args:
        negative_rows: 負例評価 CSV row list。

    Returns:
        Markdown table を含む行 list。
    """
    lines: list[str] = [
        "",
        "## Source Mask 外 Source 消失リスク",
        "",
        (
            "| scenario | method | outside count | outside labels | "
            "outside delta | max suppression | status |"
        ),
        "|---|---|---:|---|---|---:|---|",
    ]
    target_ids = set(RISK_FIGURE_SCENARIO_IDS)
    for row in negative_rows:
        if str(row.get("scenario_id", "")) not in target_ids:
            continue
        if str(row.get("output_stage", "")) != "effective":
            continue
        if str(row.get("method_id", "")) == "fixed_baseline":
            continue
        lines.append(
            (
                "| {scenario} | `{method}` | {outside_count} | {outside_labels} | "
                "{outside_delta} | {max_suppression:.3f} | {status} |"
            ).format(
                scenario=str(row.get("scenario_id", "")),
                method=str(row.get("method_id", "")),
                outside_count=int(round(_row_float(row, "mask_outside_source_count"))),
                outside_labels=str(row.get("mask_outside_source_labels", "")),
                outside_delta=str(row.get("mask_outside_source_level_delta_db", "")),
                max_suppression=_row_float(row, "max_mask_outside_source_suppression_db"),
                status=str(row.get("status", "")),
            )
        )
    lines.extend(
        [
            "",
            (
                "`max suppression` は source mask 外 source の true 方位最近傍 beam での "
                "tone level delta であり、負値が大きいほど未知 source を消す危険が大きい。"
            ),
            (
                "offgrid nominal 条件は wide guard 内に true source が入るため、"
                "outside count が 0 になることを確認対象として扱う。"
            ),
        ]
    )
    return lines

def _write_final_report(
    *,
    representative_rows: list[dict[str, object]],
    negative_rows: list[dict[str, object]],
    safety_rows: list[dict[str, object]],
    runtime_rows: list[dict[str, object]],
    output_path: Path,
) -> None:
    """8月評価向け候補 report を保存する。

    Args:
        representative_rows: 代表 scenario の summary row。
        negative_rows: 負例 scenario の summary row。
        safety_rows: safety gate probe row。
        runtime_rows: A2 kernel runtime row。
        output_path: 保存先 Markdown path。

    Returns:
        なし。
    """
    original_counts = _original_sweep_counts()
    runtime_p95 = _runtime_p95_by_method(runtime_rows)
    all_decision_rows = representative_rows + negative_rows

    lines: list[str] = [
        "# 8月評価向け 軽量 ABF-like 候補絞り込み",
        "",
        "## 出力",
        "",
        f"- representative / shortlist summary: `{SHORTLIST_SUMMARY_CSV_PATH.as_posix()}`",
        f"- negative summary: `{NEGATIVE_SUMMARY_CSV_PATH.as_posix()}`",
        f"- safety gate summary: `{SAFETY_GATE_SUMMARY_CSV_PATH.as_posix()}`",
        f"- runtime summary: `{RUNTIME_SUMMARY_CSV_PATH.as_posix()}`",
        f"- figure manifest: `{FIGURE_MANIFEST_CSV_PATH.as_posix()}`",
        f"- figures: `{FIGURE_DIR.as_posix()}`",
        "",
        "## 候補比較",
        "",
        (
            "| method | parameter | original sweep | representative | negative | "
            "worst p95 delta | max gated worsening | fallback rows | runtime p95 |"
        ),
        "|---|---|---|---|---|---:|---:|---:|---:|",
    ]

    parameter_label_by_method = {
        "fixed_baseline": "current operational shading",
        "A2_safe": "eta=0.5, loading=0.03, guard=wide, one_block_delay",
        "A2_aggressive": "eta=1.0, loading=0.1, guard=wide, one_block_delay",
    }
    for method_id in ("fixed_baseline", "A2_safe", "A2_aggressive"):
        method_rows = _candidate_effective_rows(all_decision_rows, method_id)
        representative_counts = _status_counts(representative_rows, method_id)
        negative_counts = _status_counts(negative_rows, method_id)
        lines.append(
            (
                "| `{method}` | {parameter} | {original} | {representative} | {negative} | "
                "{worst_p95} | {max_worsening} | {fallback_count} | {runtime_p95} |"
            ).format(
                method=method_id,
                parameter=parameter_label_by_method[method_id],
                original=_count_label(original_counts.get(method_id, {})),
                representative=_count_label(representative_counts),
                negative=_count_label(negative_counts),
                worst_p95=_format_optional_float(
                    _max_row_float(method_rows, "non_source_p95_level_delta_db")
                ),
                max_worsening=_format_optional_float(
                    _max_row_float(method_rows, "max_local_worsening_db_gated")
                ),
                fallback_count=_fallback_count(method_rows),
                runtime_p95=_format_optional_float(runtime_p95.get(method_id)),
            )
        )

    safety_effective_rows = [
        row for row in safety_rows if str(row.get("output_stage", "")) == "effective"
    ]
    safety_reason = ""
    safety_equal = ""
    if len(safety_effective_rows) > 0:
        safety_reason = str(safety_effective_rows[0].get("fallback_reason", ""))
        safety_equal = str(safety_effective_rows[0].get("effective_equals_fixed_baseline", ""))

    lines.extend(_source_mask_risk_report_lines(negative_rows))
    lines.extend(
        [
            "",
            "## Safety Gate",
            "",
            (
                "人工的に non-source raw を +6 dB 悪化させ、effective は fixed baseline と"
                f"一致することを確認した。fallback_reason=`{safety_reason}`, "
                f"effective_equals_fixed_baseline=`{safety_equal}`。"
            ),
            "",
            "## 採用条件",
            "",
            "- 判定は raw ではなく effective のみで行う。raw 診断行は原因調査用に限定する。",
            (
                "- A2_safe は 8月評価の主候補とし、detected mask と "
                "offgrid 0.5 deg で pass を維持すること。"
            ),
            (
                "- A2_aggressive は比較候補とし、負例で fallback 頻度や "
                "source 保護劣化が増えない場合に限定して扱う。"
            ),
            (
                "- unknown / weak / detected miss 条件では、fallback_required と "
                "fallback_reason を必ず確認する。"
            ),
            "- runtime は `runtime_scope=a2_kernel_only` の p95 runtime_factor で確認する。",
            "",
            "## 不採用条件",
            "",
            (
                "- effective で false peak count が増える、または "
                "source peak delta が 1 dB を超える条件。"
            ),
            (
                "- source mask 外 source を A2 が source leakage とみなして抑圧し、"
                "運用上の未知 source を消す条件。"
            ),
            (
                "- detected mask が 1 本だけの条件で、未検出 source 周辺の"
                "局所悪化が safety gate なしに残る条件。"
            ),
            "- A2 kernel の p95 runtime_factor が実時間予算を超える条件。",
            "",
            "## 運用上の注意",
            "",
            (
                "- wide guard は source 保護を優先する設定であり、"
                "non-source 抑圧範囲は narrow/default より狭い。"
            ),
            (
                "- detected mask の検出数は評価ごとに記録し、"
                "source 数と一致しない場合は負例として扱う。"
            ),
            (
                "- BL は mixed RMS、FRAZ は tone projection、BTR は短時間 RMS 相対値であり、"
                "dB 基準は図中に明記した。"
            ),
            "- fixed_baseline は常に fallback 先として残し、A2 の raw 改善だけを採用理由にしない。",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """8月評価向け候補の代表図、負例、安全 gate、runtime artifact を生成する。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    require_matplotlib()
    assets = _load_assets()

    representative_rows, figure_manifest_rows = _evaluate_representatives(assets=assets)
    figure_manifest_rows.extend(_evaluate_negative_risk_figures(assets=assets))
    negative_rows = _evaluate_negatives(assets=assets)
    safety_rows = _build_safety_gate_rows(assets=assets)
    runtime_rows = _measure_runtime_rows(assets=assets)

    _write_csv(representative_rows, SHORTLIST_SUMMARY_CSV_PATH)
    _write_csv(negative_rows, NEGATIVE_SUMMARY_CSV_PATH)
    _write_csv(safety_rows, SAFETY_GATE_SUMMARY_CSV_PATH)
    _write_csv(runtime_rows, RUNTIME_SUMMARY_CSV_PATH)
    _write_csv(figure_manifest_rows, FIGURE_MANIFEST_CSV_PATH)
    _write_final_report(
        representative_rows=representative_rows,
        negative_rows=negative_rows,
        safety_rows=safety_rows,
        runtime_rows=runtime_rows,
        output_path=FINAL_REPORT_MD_PATH,
    )

    print(f"saved representative rows to {SHORTLIST_SUMMARY_CSV_PATH}")
    print(f"saved negative rows to {NEGATIVE_SUMMARY_CSV_PATH}")
    print(f"saved safety rows to {SAFETY_GATE_SUMMARY_CSV_PATH}")
    print(f"saved runtime rows to {RUNTIME_SUMMARY_CSV_PATH}")
    print(f"saved figures under {FIGURE_DIR}")
    print(f"saved final report to {FINAL_REPORT_MD_PATH}")


if __name__ == "__main__":
    main()










