"""source mask 外 source の消失リスクを条件 sweep で評価する。

このスクリプトは、8月評価向けに絞り込んだ `A2_safe` / `A2_aggressive` について、
detected mask が未知または弱い source を拾えない条件を意図的に作り、source visibility が
noise floor、強い target の sidelobe、周波数近接、方位近接、source 数でどう崩れるかを保存する。

出力先は `artifacts/beamforming/lightweight_abf_like_comparison/source_mask_risk_sweep` である。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from examples.beamforming.evaluate_lightweight_abf_like_august_shortlist import (
    SHORTLIST_CANDIDATES,
    WIDE_GUARD,
    CandidateEvaluation,
    ShortlistCandidate,
    _evaluate_candidate,
    _evaluation_window,
)
from examples.beamforming.evaluate_lightweight_abf_like_comparison import (
    ARRAY_DEFINITION_PATH,
    DETECTED_SOURCE_THRESHOLD_DB_BELOW_PEAK,
    FRACTIONAL_DELAY_FILTER_BANK_PATH,
    LEVEL_UNIT_LABEL,
    N_BEAM_AZ_REAL,
    N_SAMPLE,
    SAMPLE_PER_DOF_MIN,
    SHADING_DEFINITION_PATH,
    B1Candidate,
    FrequencyResponseCacheEntry,
    SourceSpec,
    _complex_response_for_source,
    _metrics_row_fields,
    _rms_levels_db20,
    _source_mask_for_case,
    _weights_for_candidate,
    _write_csv,
)
from spflow.beamforming import SourceSectorMask
from spflow.beamforming.diagnostic_plotting import require_matplotlib
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


OUTPUT_DIR = Path(
    "artifacts/beamforming/lightweight_abf_like_comparison/source_mask_risk_sweep"
)
FIGURE_DIR = OUTPUT_DIR / "figures"
SUMMARY_CSV_PATH = OUTPUT_DIR / "source_mask_risk_sweep_summary.csv"
REPORT_MD_PATH = OUTPUT_DIR / "source_mask_risk_sweep_report.md"

TARGET_AZIMUTH_DEG = 90.0
TARGET_FREQUENCY_HZ = 10000.0
TARGET_LEVEL_DB = 0.0
TARGET_PHASE_RAD = 0.0
UNKNOWN_BASE_AZIMUTH_DEG = 60.0
UNKNOWN_BASE_FREQUENCY_HZ = 8192.0
UNKNOWN_PHASE_RAD = 1.3
NOISE_SEED_BASE = 20260706


@dataclass(frozen=True)
class RiskAssets:
    """source mask risk sweep で共有する beamforming asset を保持する。

    このクラスは、運用 sparse array、運用 shading、小数遅延 FIR bank、方位軸、
    周波数別 fixed beam response cache をまとめる。

    入力は保存済み array / shading / filter bank artifact と sweep 周波数列であり、
    出力は source 合成と detected mask 作成に使う共有状態である。

    A2 重み推定、source 検出の採否、CSV / 図保存は責務に含めない。
    信号処理上は、条件 sweep の各行で同じ固定整相応答を再利用するための前処理である。
    """

    array_definition: OperationalSparseArrayDefinition
    shading_definition: OperationalShadingDefinition
    axis_azimuth_deg: FloatArray
    response_cache: dict[float, FrequencyResponseCacheEntry]


@dataclass(frozen=True)
class RiskScenario:
    """source mask 外 source リスク確認の 1 条件を保持する。

    このクラスは、target、unknown source、追加 source、beam-domain noise floor を
    1 つの評価単位としてまとめる。

    入力は source 方位 deg、周波数 Hz、RMS レベル dB re input RMS、
    noise floor dB re input RMS であり、出力は fixed / A2 の metric 行である。

    A2 の係数推定や source mask の構築は責務に含めない。
    信号処理上は、detected mask が source を見落とす条件を明示して source visibility
    preservation を評価する scenario 定義である。
    """

    scenario_id: str
    scenario_family: str
    unknown_source: SourceSpec
    noise_floor_db: float
    extra_sources: tuple[SourceSpec, ...]
    notes: str


@dataclass(frozen=True)
class A2ComponentWeights:
    """混合信号から推定した A2 重みを source 単体へ適用するための状態を保持する。

    このクラスは、one-block-delay の前半 block で推定した non-source beam 別 SLC 重みと、
    後半 block に対応する beam index を保持する。

    入力は mixed beam output と detected source mask であり、出力は unknown source 単体の
    `y_b[n] = d_b[n] - eta h_b^H x_S[n]` を再現するための係数である。

    source mask 作成や safety gate 判定は責務に含めない。
    信号処理上は、同一周波数 source が tone projection で混ざる条件でも source 固有の
    抑圧量を分離して測るための診断状態である。
    """

    weights: ComplexArray | None
    source_reference_beams: IntArray
    non_source_beams: IntArray
    fallback_reasons: tuple[str, ...]


def _target_source() -> SourceSpec:
    """基準 target source を返す。

    Returns:
        90 deg / 10000 Hz / 0 dB re input RMS の source 条件。
    """
    return SourceSpec(
        label="target",
        frequency_hz=TARGET_FREQUENCY_HZ,
        azimuth_deg=TARGET_AZIMUTH_DEG,
        level_db=TARGET_LEVEL_DB,
        phase_rad=TARGET_PHASE_RAD,
    )


def _collect_sweep_frequencies(scenarios: tuple[RiskScenario, ...]) -> FloatArray:
    """sweep に必要な周波数一覧を返す。

    Args:
        scenarios: source risk scenario 列。

    Returns:
        一意な周波数列。shape は `[n_freq]`、単位は Hz。
    """
    frequencies = {TARGET_FREQUENCY_HZ}
    for scenario in scenarios:
        frequencies.add(float(scenario.unknown_source.frequency_hz))
        for source in scenario.extra_sources:
            frequencies.add(float(source.frequency_hz))
    return np.asarray(sorted(frequencies), dtype=np.float64)


def _risk_scenarios() -> tuple[RiskScenario, ...]:
    """source mask risk sweep の条件列を作る。

    Returns:
        レベル/SNR、周波数近接、方位近接、source 数を振った scenario tuple。

    境界条件:
        detected mask は最大 2 peak 検出であり、weak source が閾値下または sidelobe に埋もれると
        mask 外 source として扱われる。ここでは target は常に 90 deg / 10000 Hz / 0 dB とし、
        unknown/weak source の条件だけを振る。
    """
    scenarios: list[RiskScenario] = []

    for unknown_level_db in (-6.0, -12.0, -18.0, -24.0, -30.0, -36.0):
        for noise_floor_db in (-70.0, -60.0, -50.0, -40.0, -35.0, -30.0):
            scenario_id = (
                "risk_level_snr_"
                f"u{_format_signed_db(unknown_level_db)}_n{_format_signed_db(noise_floor_db)}"
            )
            scenarios.append(
                RiskScenario(
                    scenario_id=scenario_id,
                    scenario_family="level_snr",
                    unknown_source=SourceSpec(
                        label="unknown_60deg",
                        frequency_hz=UNKNOWN_BASE_FREQUENCY_HZ,
                        azimuth_deg=UNKNOWN_BASE_AZIMUTH_DEG,
                        level_db=unknown_level_db,
                        phase_rad=UNKNOWN_PHASE_RAD,
                    ),
                    noise_floor_db=noise_floor_db,
                    extra_sources=(),
                    notes="60 deg unknown source のレベルと beam-domain noise floor を振る。",
                )
            )

    for unknown_frequency_hz in (6144.0, 8192.0, 9500.0, 9800.0, 10000.0):
        scenario_id = f"risk_frequency_proximity_f{int(round(unknown_frequency_hz))}"
        scenarios.append(
            RiskScenario(
                scenario_id=scenario_id,
                scenario_family="frequency_proximity",
                unknown_source=SourceSpec(
                    label="unknown_frequency_proximity",
                    frequency_hz=unknown_frequency_hz,
                    azimuth_deg=UNKNOWN_BASE_AZIMUTH_DEG,
                    level_db=-24.0,
                    phase_rad=UNKNOWN_PHASE_RAD,
                ),
                noise_floor_db=-50.0,
                extra_sources=(),
                notes="60 deg unknown source の周波数を target へ近づける。",
            )
        )

    for unknown_azimuth_deg in (60.0, 75.0, 82.5, 86.0, 88.0, 92.0):
        scenario_id = f"risk_azimuth_proximity_a{_format_azimuth(unknown_azimuth_deg)}"
        scenarios.append(
            RiskScenario(
                scenario_id=scenario_id,
                scenario_family="azimuth_proximity",
                unknown_source=SourceSpec(
                    label="unknown_azimuth_proximity",
                    frequency_hz=UNKNOWN_BASE_FREQUENCY_HZ,
                    azimuth_deg=unknown_azimuth_deg,
                    level_db=-24.0,
                    phase_rad=UNKNOWN_PHASE_RAD,
                ),
                noise_floor_db=-50.0,
                extra_sources=(),
                notes="unknown source 方位を target source mask へ近づける。",
            )
        )

    extra_source_sets: tuple[tuple[SourceSpec, ...], ...] = (
        (),
        (
            SourceSpec(
                label="extra_120deg",
                frequency_hz=6144.0,
                azimuth_deg=120.0,
                level_db=-12.0,
                phase_rad=2.1,
            ),
        ),
        (
            SourceSpec(
                label="extra_120deg",
                frequency_hz=6144.0,
                azimuth_deg=120.0,
                level_db=-12.0,
                phase_rad=2.1,
            ),
            SourceSpec(
                label="extra_150deg",
                frequency_hz=9500.0,
                azimuth_deg=150.0,
                level_db=-18.0,
                phase_rad=2.7,
            ),
        ),
        (
            SourceSpec(
                label="extra_45deg",
                frequency_hz=6144.0,
                azimuth_deg=45.0,
                level_db=-9.0,
                phase_rad=1.9,
            ),
            SourceSpec(
                label="extra_120deg",
                frequency_hz=8192.0,
                azimuth_deg=120.0,
                level_db=-15.0,
                phase_rad=2.4,
            ),
            SourceSpec(
                label="extra_150deg",
                frequency_hz=9500.0,
                azimuth_deg=150.0,
                level_db=-18.0,
                phase_rad=2.9,
            ),
        ),
    )
    for source_count_index, extra_sources in enumerate(extra_source_sets):
        true_source_count = 2 + len(extra_sources)
        scenarios.append(
            RiskScenario(
                scenario_id=f"risk_source_count_{true_source_count}",
                scenario_family="source_count",
                unknown_source=SourceSpec(
                    label="unknown_source_count",
                    frequency_hz=UNKNOWN_BASE_FREQUENCY_HZ,
                    azimuth_deg=UNKNOWN_BASE_AZIMUTH_DEG,
                    level_db=-24.0,
                    phase_rad=UNKNOWN_PHASE_RAD + 0.1 * float(source_count_index),
                ),
                noise_floor_db=-50.0,
                extra_sources=extra_sources,
                notes="true source 数を増やし、detected mask の最大 2 peak 制約を確認する。",
            )
        )

    return tuple(scenarios)


def _format_signed_db(value_db: float) -> str:
    """ファイル名に使える signed dB 表現を返す。"""
    sign = "m" if float(value_db) < 0.0 else "p"
    return f"{sign}{abs(float(value_db)):04.1f}".replace(".", "p")


def _format_azimuth(azimuth_deg: float) -> str:
    """ファイル名に使える方位表現を返す。"""
    return f"{float(azimuth_deg):05.1f}".replace(".", "p")


def _load_risk_assets(frequencies_hz: FloatArray) -> RiskAssets:
    """評価 artifact を読み込み、必要周波数の fixed beam response cache を作る。

    Args:
        frequencies_hz: sweep で使う周波数列。shape は `[n_freq]`、単位は Hz。

    Returns:
        source 合成に使う共有 asset。

    Raises:
        FileNotFoundError: 必要 artifact が存在しない場合。
        ValueError: 周波数列が空の場合。
    """
    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    if frequencies.ndim != 1 or frequencies.size == 0:
        raise ValueError("frequencies_hz must have shape (n_freq,).")

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
    for frequency_hz in frequencies.tolist():
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

    return RiskAssets(
        array_definition=array_definition,
        shading_definition=shading_definition,
        axis_azimuth_deg=axis_azimuth_deg,
        response_cache=response_cache,
    )


def _operational_weights_for_sources(
    *,
    assets: RiskAssets,
    source_specs: tuple[SourceSpec, ...],
) -> dict[float, FloatArray]:
    """source 群に対する運用 shading weight を周波数別に返す。

    Args:
        assets: 配列・shading・周波数応答 cache。
        source_specs: 合成する source 条件列。shape は `[n_source]`。

    Returns:
        周波数 Hz を key とする channel weight 辞書。
        各 value の shape は `[n_active_ch_at_frequency]`。
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
    assets: RiskAssets,
    weights_by_frequency: dict[float, FloatArray],
    n_sample: int,
) -> ComplexArray:
    """複数 source の fixed beam-domain 波形を合成する。

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
        # この sweep は表示系の相対比較であり、既存 comparison と同じ dB re input RMS に揃える。
        tone = amplitude_rms * np.exp(
            1j
            * (
                2.0 * np.pi * frequency_hz * time_axis_s
                + float(source_spec.phase_rad)
            )
        )
        # response[:, None] shape: [n_beam, 1]、tone[None, :] shape: [1, n_sample]。
        # broadcasting で beam 応答と時間波形を掛け、source ごとに加算する。
        beam_output += response[:, np.newaxis] * tone[np.newaxis, :]

    return np.asarray(beam_output, dtype=np.complex128)


def _beam_domain_noise(
    *,
    n_beam: int,
    n_sample: int,
    noise_floor_db: float,
    seed: int,
) -> ComplexArray:
    """beam-domain の表示 noise floor を合成する。

    Args:
        n_beam: beam 本数。
        n_sample: sample 数。
        noise_floor_db: 複素 RMS noise level。単位は dB re input RMS。
        seed: 再現性のための乱数 seed。

    Returns:
        複素 white noise。shape は `[n_beam, n_sample]`。

    境界条件:
        ここでは channel-domain の相関 noise ではなく、BL 表示上の floor と検出閾値の関係を
        分離して見るため beam-domain に加える。空間相関 noise の評価は別条件で扱う必要がある。
    """
    if int(n_beam) <= 0 or int(n_sample) <= 0:
        raise ValueError("n_beam and n_sample must be positive.")
    rng = np.random.default_rng(int(seed))
    amplitude_rms = float(10.0 ** (float(noise_floor_db) / 20.0))
    # (real + j imag) / sqrt(2) により、複素振幅の RMS が amplitude_rms になる。
    noise = (
        rng.standard_normal((int(n_beam), int(n_sample)))
        + 1j * rng.standard_normal((int(n_beam), int(n_sample)))
    ) * (amplitude_rms / np.sqrt(2.0))
    return np.asarray(noise, dtype=np.complex128)


def _source_mask_for_detected(
    *,
    assets: RiskAssets,
    before_levels_db: FloatArray,
) -> SourceSectorMask:
    """detected mask を mixed BL から作る。

    Args:
        assets: 方位軸を含む共有 asset。
        before_levels_db: fixed baseline の BL。shape は `[n_beam]`、単位は dB re input RMS。

    Returns:
        detected source / non-source sector mask。

    境界条件:
        source_azimuths_deg は detected mask では使われないが、既存 helper の signature を満たすため
        target 方位を渡す。実際の mask は BL peak 検出だけで決まる。
    """
    return _source_mask_for_case(
        axis_azimuth_deg=assets.axis_azimuth_deg,
        before_levels_db=before_levels_db,
        source_azimuths_deg=np.asarray([TARGET_AZIMUTH_DEG], dtype=np.float64),
        guard=WIDE_GUARD,
        mask_type="detected",
    )


def _estimate_component_weights(
    *,
    mixed_output: ComplexArray,
    source_mask: SourceSectorMask,
    source_reference_beams: IntArray,
    loading: float,
) -> A2ComponentWeights:
    """mixed 出力から one-block-delay A2 重みを推定する。

    Args:
        mixed_output: noise を含む fixed baseline mixed 波形。shape は `[n_beam, n_sample]`。
        source_mask: detected source mask。mask shape は `[n_beam]`。
        source_reference_beams: source reference beam index。shape は `[n_ref]`。
        loading: relative diagonal loading。無次元。

    Returns:
        unknown source 単体へ同じ A2 重みを適用するための状態。

    信号処理式:
        前半 block の source reference `X_S` と non-source `d_b` から
        `R_ss = X_S X_S^H / K`、`r_sd = d_b^H X_S / K` を作り、
        `(R_ss + λ mean(diag(R_ss)) I) h_b = r_sd` を解く。
        後段では `h_b^H x_S[n]` を source-correlated leakage として差し引く。
    """
    signals = np.asarray(mixed_output, dtype=np.complex128)
    if signals.ndim != 2:
        raise ValueError("mixed_output must have shape (n_beam, n_sample).")

    split_index = int(signals.shape[1] // 2)
    if split_index <= 0 or split_index >= signals.shape[1]:
        raise ValueError("one-block-delay requires at least two samples.")

    reference_indices = np.asarray(source_reference_beams, dtype=np.int64)
    non_source_beams = np.flatnonzero(source_mask.non_source_mask).astype(np.int64)
    n_ref = int(reference_indices.size)
    if n_ref <= 0 or non_source_beams.size == 0:
        return A2ComponentWeights(
            weights=None,
            source_reference_beams=reference_indices,
            non_source_beams=non_source_beams,
            fallback_reasons=("component_weight_reference_or_non_source_empty",),
        )

    train_signals = signals[:, :split_index]
    n_train_sample = int(train_signals.shape[1])
    source_reference_train = train_signals[reference_indices, :]
    train_non_source = train_signals[non_source_beams, :]
    covariance_matrix = np.asarray(
        (source_reference_train @ source_reference_train.conj().T) / float(n_train_sample),
        dtype=np.complex128,
    )
    average_power = float(np.real(np.trace(covariance_matrix)) / float(n_ref))
    if not bool(np.isfinite(average_power)) or average_power <= 0.0:
        # source reference が極小の場合でも loaded covariance を構成し、失敗理由を CSV に残す。
        average_power = 1.0
    loading_power = float(loading) * average_power
    loaded_covariance = covariance_matrix + loading_power * np.eye(n_ref, dtype=np.complex128)
    cross_correlations = np.asarray(
        (train_non_source.conj() @ source_reference_train.T) / float(n_train_sample),
        dtype=np.complex128,
    )

    try:
        weights = np.asarray(
            np.linalg.solve(loaded_covariance, cross_correlations.T).T,
            dtype=np.complex128,
        )
    except np.linalg.LinAlgError:
        return A2ComponentWeights(
            weights=None,
            source_reference_beams=reference_indices,
            non_source_beams=non_source_beams,
            fallback_reasons=("component_weight_linear_solve_failed",),
        )

    return A2ComponentWeights(
        weights=weights,
        source_reference_beams=reference_indices,
        non_source_beams=non_source_beams,
        fallback_reasons=(),
    )


def _apply_component_a2(
    *,
    component_output: ComplexArray,
    source_mask: SourceSectorMask,
    component_weights: A2ComponentWeights,
    eta: float,
    fallback_to_fixed: bool,
) -> ComplexArray:
    """推定済み A2 重みを source 単体の後半 block へ適用する。

    Args:
        component_output: unknown source 単体の fixed beam 波形。shape は `[n_beam, n_sample]`。
        source_mask: detected source mask。mask shape は `[n_beam]`。
        component_weights: mixed 信号から推定した A2 重み。
        eta: キャンセル量係数。無次元。
        fallback_to_fixed: effective 出力が fixed baseline へ戻る場合は True。

    Returns:
        source 単体に対する effective 相当の後半 block。shape は `[n_beam, n_sample / 2]`。

    境界条件:
        mixed 側で fallback した場合、運用出力は fixed baseline なので source 単体も後半 block を
        そのまま返す。これにより raw ではなく effective の source 消失リスクを測る。
    """
    signals = np.asarray(component_output, dtype=np.complex128)
    if signals.ndim != 2:
        raise ValueError("component_output must have shape (n_beam, n_sample).")
    split_index = int(signals.shape[1] // 2)
    eval_signals = signals[:, split_index:].copy()

    if fallback_to_fixed or component_weights.weights is None:
        return np.asarray(eval_signals, dtype=np.complex128)

    non_source_beams = component_weights.non_source_beams
    reference_indices = component_weights.source_reference_beams
    source_reference_eval = eval_signals[reference_indices, :]
    # weights shape: [n_non_source, n_ref]、source_reference_eval shape: [n_ref, n_eval_sample]。
    # conj(weights) @ X_S により、各 non-source beam へ漏れる source-correlated 成分を推定する。
    cancel_estimate = np.conj(component_weights.weights) @ source_reference_eval
    eval_signals[non_source_beams, :] = (
        eval_signals[non_source_beams, :] - float(eta) * cancel_estimate
    )
    eval_signals[source_mask.source_mask, :] = signals[source_mask.source_mask, split_index:]
    return np.asarray(eval_signals, dtype=np.complex128)


def _source_flags(
    *,
    source_specs: tuple[SourceSpec, ...],
    source_mask: SourceSectorMask,
    axis_azimuth_deg: FloatArray,
) -> tuple[str, str]:
    """source ごとの mask 内外 flag を文字列化する。

    Args:
        source_specs: true source 条件列。shape は `[n_source]`。
        source_mask: detected source mask。
        axis_azimuth_deg: beam 方位軸。shape は `[n_beam]`、単位は deg。

    Returns:
        `(labels, flags)`。flags は `true|false` 形式で、nearest beam が mask 内なら true。
    """
    axis = np.asarray(axis_azimuth_deg, dtype=np.float64)
    labels: list[str] = []
    flags: list[str] = []
    for source in source_specs:
        nearest_beam = int(np.argmin(np.abs(axis - float(source.azimuth_deg))))
        labels.append(str(source.label))
        flags.append("true" if bool(source_mask.source_mask[nearest_beam]) else "false")
    return "|".join(labels), "|".join(flags)


def _single_beam_level_db(output: ComplexArray, beam_index: int) -> float:
    """指定 beam の RMS レベルを dB re input RMS で返す。"""
    signals = np.asarray(output, dtype=np.complex128)
    if signals.ndim != 2:
        raise ValueError("output must have shape (n_beam, n_sample).")
    rms = float(np.sqrt(np.mean(np.abs(signals[int(beam_index), :]) ** 2)))
    return float(20.0 * np.log10(max(rms, np.finfo(np.float64).tiny)))


def _evaluate_risk_scenario(
    *,
    scenario: RiskScenario,
    assets: RiskAssets,
    seed_offset: int,
) -> list[dict[str, object]]:
    """1 scenario を fixed / A2_safe / A2_aggressive で評価する。

    Args:
        scenario: source mask risk scenario。
        assets: beam-domain 合成に使う共有 asset。
        seed_offset: noise 生成 seed の offset。

    Returns:
        summary CSV row list。
    """
    target = _target_source()
    source_specs = (target, scenario.unknown_source) + scenario.extra_sources
    weights_by_frequency = _operational_weights_for_sources(
        assets=assets,
        source_specs=source_specs,
    )
    target_output = _synthesize_multisource_beam_output(
        source_specs=(target,),
        assets=assets,
        weights_by_frequency=weights_by_frequency,
        n_sample=N_SAMPLE,
    )
    unknown_output = _synthesize_multisource_beam_output(
        source_specs=(scenario.unknown_source,),
        assets=assets,
        weights_by_frequency=weights_by_frequency,
        n_sample=N_SAMPLE,
    )
    mixed_without_noise = _synthesize_multisource_beam_output(
        source_specs=source_specs,
        assets=assets,
        weights_by_frequency=weights_by_frequency,
        n_sample=N_SAMPLE,
    )
    noise = _beam_domain_noise(
        n_beam=int(mixed_without_noise.shape[0]),
        n_sample=int(mixed_without_noise.shape[1]),
        noise_floor_db=float(scenario.noise_floor_db),
        seed=NOISE_SEED_BASE + int(seed_offset),
    )
    mixed_output = np.asarray(mixed_without_noise + noise, dtype=np.complex128)
    before_levels_db = _rms_levels_db20(mixed_output)
    source_mask = _source_mask_for_detected(
        assets=assets,
        before_levels_db=before_levels_db,
    )
    source_reference_beams = np.flatnonzero(source_mask.source_mask).astype(np.int64)
    evaluation_duration_s = float(_evaluation_window(mixed_output).shape[1]) / float(
        assets.array_definition.fs_hz
    )

    axis = np.asarray(assets.axis_azimuth_deg, dtype=np.float64)
    unknown_beam = int(np.argmin(np.abs(axis - float(scenario.unknown_source.azimuth_deg))))
    source_labels, source_visibility_flags = _source_flags(
        source_specs=source_specs,
        source_mask=source_mask,
        axis_azimuth_deg=axis,
    )
    unknown_in_mask = bool(source_mask.source_mask[unknown_beam])
    detected_source_count = int(source_mask.source_beam_indices.size)
    true_source_count = int(len(source_specs))
    unknown_before_window = _evaluation_window(unknown_output)
    unknown_before_level_db = _single_beam_level_db(unknown_before_window, unknown_beam)
    noise_window = _evaluation_window(noise)
    noise_level_at_unknown_db = _single_beam_level_db(noise_window, unknown_beam)
    target_window = _evaluation_window(target_output)
    target_sidelobe_at_unknown_db = _single_beam_level_db(target_window, unknown_beam)
    mixed_unknown_bl_level_db = float(before_levels_db[unknown_beam])
    global_peak_level_db = float(np.max(before_levels_db))
    detector_threshold_level_db = float(
        global_peak_level_db - DETECTED_SOURCE_THRESHOLD_DB_BELOW_PEAK
    )
    margin_to_detector_threshold_db = float(
        mixed_unknown_bl_level_db - detector_threshold_level_db
    )
    margin_over_noise_db = float(unknown_before_level_db - noise_level_at_unknown_db)
    margin_over_target_sidelobe_db = float(
        unknown_before_level_db - target_sidelobe_at_unknown_db
    )

    rows: list[dict[str, object]] = []
    for candidate in SHORTLIST_CANDIDATES:
        evaluation = _evaluate_candidate(
            candidate=candidate,
            before_output=mixed_output,
            source_mask=source_mask,
            source_reference_beams=source_reference_beams,
            axis_azimuth_deg=axis,
            evaluation_duration_s=evaluation_duration_s,
        )
        rows.append(
            _risk_row(
                scenario=scenario,
                candidate=candidate,
                evaluation=evaluation,
                assets=assets,
                source_mask=source_mask,
                source_labels=source_labels,
                source_visibility_flags=source_visibility_flags,
                unknown_beam=unknown_beam,
                unknown_before_window=unknown_before_window,
                unknown_output=unknown_output,
                mixed_output=mixed_output,
                unknown_in_mask=unknown_in_mask,
                detected_source_count=detected_source_count,
                true_source_count=true_source_count,
                unknown_before_level_db=unknown_before_level_db,
                noise_level_at_unknown_db=noise_level_at_unknown_db,
                target_sidelobe_at_unknown_db=target_sidelobe_at_unknown_db,
                mixed_unknown_bl_level_db=mixed_unknown_bl_level_db,
                detector_threshold_level_db=detector_threshold_level_db,
                margin_to_detector_threshold_db=margin_to_detector_threshold_db,
                margin_over_noise_db=margin_over_noise_db,
                margin_over_target_sidelobe_db=margin_over_target_sidelobe_db,
            )
        )
    return rows


def _risk_row(
    *,
    scenario: RiskScenario,
    candidate: ShortlistCandidate,
    evaluation: CandidateEvaluation,
    assets: RiskAssets,
    source_mask: SourceSectorMask,
    source_labels: str,
    source_visibility_flags: str,
    unknown_beam: int,
    unknown_before_window: ComplexArray,
    unknown_output: ComplexArray,
    mixed_output: ComplexArray,
    unknown_in_mask: bool,
    detected_source_count: int,
    true_source_count: int,
    unknown_before_level_db: float,
    noise_level_at_unknown_db: float,
    target_sidelobe_at_unknown_db: float,
    mixed_unknown_bl_level_db: float,
    detector_threshold_level_db: float,
    margin_to_detector_threshold_db: float,
    margin_over_noise_db: float,
    margin_over_target_sidelobe_db: float,
) -> dict[str, object]:
    """summary CSV の 1 行を作る。

    Args:
        scenario: source mask risk scenario。
        candidate: 評価候補。
        evaluation: mixed 出力での fixed / A2 評価結果。
        assets: 方位軸と配列条件を含む共有 asset。
        source_mask: detected source mask。
        source_labels: true source label の `|` 連結文字列。
        source_visibility_flags: true source の mask 内外 flag。
        unknown_beam: unknown source 最近傍 beam index。
        unknown_before_window: unknown source 単体 fixed 後半 block。shape は `[n_beam, n_eval]`。
        unknown_output: unknown source 単体 fixed 全 block。shape は `[n_beam, n_sample]`。
        mixed_output: noise を含む mixed fixed 全 block。shape は `[n_beam, n_sample]`。
        unknown_in_mask: unknown 最近傍 beam が source mask 内なら True。
        detected_source_count: detected mask が持つ source peak 数。
        true_source_count: scenario の true source 数。
        unknown_before_level_db: unknown 単体の fixed level。単位は dB re input RMS。
        noise_level_at_unknown_db: unknown beam の noise floor。単位は dB re input RMS。
        target_sidelobe_at_unknown_db: target-only sidelobe level。単位は dB re input RMS。
        mixed_unknown_bl_level_db: mixed BL の unknown beam level。単位は dB re input RMS。
        detector_threshold_level_db: detected mask の level threshold。単位は dB re input RMS。
        margin_to_detector_threshold_db: unknown beam の threshold 超過量。単位は dB。
        margin_over_noise_db: unknown source と noise floor の差。単位は dB。
        margin_over_target_sidelobe_db: unknown source と target sidelobe の差。単位は dB。

    Returns:
        summary CSV row。
    """
    before_window_levels_db = _rms_levels_db20(evaluation.before_window)
    after_levels_db = _rms_levels_db20(evaluation.effective_output)
    row = _metrics_row_fields(
        before_levels_db=before_window_levels_db,
        after_levels_db=after_levels_db,
        axis_azimuth_deg=assets.axis_azimuth_deg,
        source_mask=source_mask,
        realtime_factor=evaluation.realtime_factor,
        nan_inf_count=int(np.count_nonzero(~np.isfinite(evaluation.effective_output))),
        condition_number=evaluation.condition_number,
    )

    if candidate.is_baseline:
        unknown_component_delta_db = 0.0
        component_weight_reasons = ""
    else:
        if candidate.eta is None or candidate.loading is None:
            raise ValueError("A2 candidate requires eta and loading.")
        component_weights = _estimate_component_weights(
            mixed_output=mixed_output,
            source_mask=source_mask,
            source_reference_beams=np.flatnonzero(source_mask.source_mask).astype(np.int64),
            loading=float(candidate.loading),
        )
        unknown_after_component = _apply_component_a2(
            component_output=unknown_output,
            source_mask=source_mask,
            component_weights=component_weights,
            eta=float(candidate.eta),
            fallback_to_fixed=bool(evaluation.fallback_required),
        )
        unknown_after_level_db = _single_beam_level_db(unknown_after_component, unknown_beam)
        unknown_component_delta_db = float(unknown_after_level_db - unknown_before_level_db)
        component_weight_reasons = "|".join(component_weights.fallback_reasons)

    if unknown_in_mask:
        mask_outside_delta_db = ""
        max_mask_outside_suppression_db = 0.0
    else:
        mask_outside_delta_db = f"{unknown_component_delta_db:.6f}"
        max_mask_outside_suppression_db = float(min(0.0, unknown_component_delta_db))

    row.update(
        {
            "scenario_id": scenario.scenario_id,
            "scenario_family": scenario.scenario_family,
            "scenario_notes": scenario.notes,
            "method_id": candidate.method_id,
            "candidate_id": candidate.candidate_id,
            "output_stage": "effective",
            "decision_used": True,
            "mask_type": "detected",
            "level_unit_label": LEVEL_UNIT_LABEL,
            "target_azimuth_deg": TARGET_AZIMUTH_DEG,
            "target_frequency_hz": TARGET_FREQUENCY_HZ,
            "target_level_db": TARGET_LEVEL_DB,
            "unknown_azimuth_deg": float(scenario.unknown_source.azimuth_deg),
            "unknown_frequency_hz": float(scenario.unknown_source.frequency_hz),
            "unknown_level_db": float(scenario.unknown_source.level_db),
            "unknown_target_azimuth_separation_deg": abs(
                float(scenario.unknown_source.azimuth_deg) - TARGET_AZIMUTH_DEG
            ),
            "unknown_target_frequency_delta_hz": abs(
                float(scenario.unknown_source.frequency_hz) - TARGET_FREQUENCY_HZ
            ),
            "noise_floor_db": float(scenario.noise_floor_db),
            "true_source_count": true_source_count,
            "detected_source_count": detected_source_count,
            "detected_source_count_mismatch": bool(detected_source_count != true_source_count),
            "source_visibility_labels": source_labels,
            "source_visibility_in_mask_flags": source_visibility_flags,
            "unknown_in_source_mask": bool(unknown_in_mask),
            "unknown_nearest_beam_index": int(unknown_beam),
            "unknown_nearest_azimuth_deg": float(assets.axis_azimuth_deg[int(unknown_beam)]),
            "unknown_fixed_component_level_db": float(unknown_before_level_db),
            "unknown_noise_floor_at_beam_db": float(noise_level_at_unknown_db),
            "target_sidelobe_at_unknown_beam_db": float(target_sidelobe_at_unknown_db),
            "mixed_unknown_beam_bl_level_db": float(mixed_unknown_bl_level_db),
            "detector_threshold_level_db": float(detector_threshold_level_db),
            "unknown_margin_to_detector_threshold_db": float(
                margin_to_detector_threshold_db
            ),
            "unknown_margin_over_noise_db": float(margin_over_noise_db),
            "unknown_margin_over_target_sidelobe_db": float(
                margin_over_target_sidelobe_db
            ),
            "unknown_component_delta_db": float(unknown_component_delta_db),
            "mask_outside_source_count": 0 if unknown_in_mask else 1,
            "mask_outside_source_labels": "" if unknown_in_mask else scenario.unknown_source.label,
            "mask_outside_source_level_delta_db": mask_outside_delta_db,
            "max_mask_outside_source_suppression_db": max_mask_outside_suppression_db,
            "eta": "" if candidate.eta is None else float(candidate.eta),
            "loading": "" if candidate.loading is None else float(candidate.loading),
            "train_mode": candidate.train_mode,
            "sample_per_dof_min": "" if candidate.is_baseline else float(SAMPLE_PER_DOF_MIN),
            "condition_number": ""
            if evaluation.condition_number is None
            else float(evaluation.condition_number),
            "weight_norm": ""
            if evaluation.weight_norm is None
            else float(evaluation.weight_norm),
            "realtime_factor": float(evaluation.realtime_factor),
            "fallback_required": bool(evaluation.fallback_required),
            "fallback_reason": "|".join(evaluation.fallback_reasons),
            "component_weight_fallback_reason": component_weight_reasons,
        }
    )
    if candidate.is_baseline:
        row["status"] = "baseline"
    return row


def _a2_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """A2 effective 行だけを返す。"""
    return [row for row in rows if str(row.get("method_id", "")).startswith("A2_")]


def _safe_float(row: dict[str, object], key: str) -> float:
    """CSV row 値を float へ変換する。空値は NaN にする。"""
    value = row.get(key, "")
    if value == "":
        return float("nan")
    if not isinstance(value, int | float | str):
        raise TypeError(f"{key} must be numeric or a numeric string.")
    return float(value)


def _write_report(rows: list[dict[str, object]]) -> None:
    """risk sweep report を Markdown で保存する。

    Args:
        rows: summary CSV と同じ row list。
    """
    a2_rows = _a2_rows(rows)
    worst_suppression_rows = sorted(
        a2_rows,
        key=lambda row: _safe_float(row, "unknown_component_delta_db"),
    )[:10]
    detector_miss_rows = [
        row
        for row in a2_rows
        if str(row.get("unknown_in_source_mask", "")) == "False"
        or row.get("unknown_in_source_mask") is False
    ]
    detector_miss_count = len(detector_miss_rows)
    total_a2_count = len(a2_rows)
    fallback_count = sum(1 for row in a2_rows if bool(row.get("fallback_required", False)))

    family_lines: list[str] = []
    for family in sorted({str(row.get("scenario_family", "")) for row in a2_rows}):
        family_rows = [row for row in a2_rows if str(row.get("scenario_family", "")) == family]
        deltas = np.asarray(
            [_safe_float(row, "unknown_component_delta_db") for row in family_rows],
            dtype=np.float64,
        )
        miss_count = sum(
            1
            for row in family_rows
            if str(row.get("unknown_in_source_mask", "")) == "False"
            or row.get("unknown_in_source_mask") is False
        )
        family_lines.append(
            "| {family} | {count} | {miss} | {min_delta:.3f} | {p05:.3f} | {median:.3f} |".format(
                family=family,
                count=len(family_rows),
                miss=miss_count,
                min_delta=float(np.nanmin(deltas)),
                p05=float(np.nanpercentile(deltas, 5.0)),
                median=float(np.nanmedian(deltas)),
            )
        )

    lines = [
        "# Source Mask Risk Sweep Report",
        "",
        "## 目的",
        "",
        (
            "detected mask が 60 deg 近傍の unknown/weak source を拾えない条件で、"
            "A2 effective 出力が source mask 外 source を消すかを確認した。"
        ),
        "",
        "## 評価条件",
        "",
        f"- target: {TARGET_AZIMUTH_DEG:.1f} deg / {TARGET_FREQUENCY_HZ:.0f} Hz / {TARGET_LEVEL_DB:.1f} dB re input RMS",
        "- mask: detected のみ",
        "- candidate: fixed_baseline / A2_safe / A2_aggressive",
        "- 判定対象: raw ではなく effective",
        (
            "- unknown source 固有の抑圧量は、mixed 信号で学習した A2 重みを "
            "unknown source 単体へ適用した `unknown_component_delta_db` で読む。"
        ),
        (
            "- noise floor は beam-domain 表示 floor と検出閾値の関係を見るための合成 noise であり、"
            "channel-domain 相関 noise ではない。"
        ),
        "",
        "## 概要",
        "",
        f"- A2 行数: {total_a2_count}",
        f"- unknown が source mask 外だった A2 行数: {detector_miss_count}",
        f"- fallback 行数: {fallback_count}",
        "",
        "## Family 別 summary",
        "",
        "| family | A2 rows | unknown outside mask rows | min unknown_component_delta_db | p05 | median |",
        "|---|---:|---:|---:|---:|---:|",
        *family_lines,
        "",
        "## Worst unknown source suppression top 10",
        "",
        "| scenario | method | unknown level | noise | freq delta | az sep | margin noise | margin sidelobe | in mask | delta | fallback |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|",
    ]
    for row in worst_suppression_rows:
        lines.append(
            "| {scenario} | {method} | {level:.1f} | {noise:.1f} | {fd:.0f} | {az:.1f} | {mn:.2f} | {ms:.2f} | {in_mask} | {delta:.3f} | {fallback} |".format(
                scenario=row.get("scenario_id", ""),
                method=row.get("method_id", ""),
                level=_safe_float(row, "unknown_level_db"),
                noise=_safe_float(row, "noise_floor_db"),
                fd=_safe_float(row, "unknown_target_frequency_delta_hz"),
                az=_safe_float(row, "unknown_target_azimuth_separation_deg"),
                mn=_safe_float(row, "unknown_margin_over_noise_db"),
                ms=_safe_float(row, "unknown_margin_over_target_sidelobe_db"),
                in_mask=row.get("unknown_in_source_mask", ""),
                delta=_safe_float(row, "unknown_component_delta_db"),
                fallback=row.get("fallback_required", ""),
            )
        )

    lines.extend(
        [
            "",
            "## 読み方",
            "",
            (
                "- `unknown_margin_to_detector_threshold_db < 0` は、BL level だけを見る detected mask では"
                " unknown source が閾値下にいることを示す。"
            ),
            (
                "- `unknown_margin_over_target_sidelobe_db < 0` は、unknown source 単体より target sidelobe が"
                "強く、表示上の peak が強信号の sidelobe に埋もれやすいことを示す。"
            ),
            (
                "- `unknown_component_delta_db` が負に大きいほど、A2 effective が mask 外 source を"
                "抑圧している。採否はこの列と `fallback_required` を raw ではなく effective として読む。"
            ),
        ]
    )
    REPORT_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_level_snr_heatmap(rows: list[dict[str, object]]) -> None:
    """level/SNR sweep の unknown component delta heatmap を保存する。

    Args:
        rows: summary CSV と同じ row list。
    """
    plt = require_matplotlib()
    family_rows = [
        row
        for row in _a2_rows(rows)
        if str(row.get("scenario_family", "")) == "level_snr"
    ]
    methods = ("A2_safe", "A2_aggressive")
    levels = sorted({_safe_float(row, "unknown_level_db") for row in family_rows})
    noises = sorted({_safe_float(row, "noise_floor_db") for row in family_rows})
    fig, axes = plt.subplots(1, len(methods), figsize=(12, 4.5), sharex=True, sharey=True)
    image = None
    for axis_index, method in enumerate(methods):
        ax = axes[axis_index]
        grid = np.full((len(noises), len(levels)), np.nan, dtype=np.float64)
        for row in family_rows:
            if str(row.get("method_id", "")) != method:
                continue
            noise_index = noises.index(_safe_float(row, "noise_floor_db"))
            level_index = levels.index(_safe_float(row, "unknown_level_db"))
            grid[noise_index, level_index] = _safe_float(row, "unknown_component_delta_db")
        image = ax.imshow(
            grid,
            origin="lower",
            aspect="auto",
            extent=[
                min(levels) - 3.0,
                max(levels) + 3.0,
                min(noises) - 5.0,
                max(noises) + 5.0,
            ],
            vmin=-6.0,
            vmax=1.0,
            cmap="coolwarm",
        )
        ax.set_title(method)
        ax.set_xlabel("unknown source level [dB re input RMS]")
        if axis_index == 0:
            ax.set_ylabel("beam-domain noise floor [dB re input RMS]")
        ax.grid(color="black", alpha=0.15, linewidth=0.5)
    if image is None:
        raise RuntimeError("heatmap requires at least one method.")
    fig.colorbar(image, ax=axes.ravel().tolist(), label="unknown component delta [dB]")
    fig.suptitle("level/SNR sweep: A2 effective unknown source preservation")
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / "level_snr_unknown_component_delta.png", dpi=150)
    plt.close(fig)


def _plot_margin_scatter(rows: list[dict[str, object]]) -> None:
    """検出閾値 margin と unknown component delta の散布図を保存する。

    Args:
        rows: summary CSV と同じ row list。
    """
    plt = require_matplotlib()
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    colors = {"A2_safe": "#1f77b4", "A2_aggressive": "#ff7f0e"}
    for method, color in colors.items():
        method_rows = [row for row in _a2_rows(rows) if str(row.get("method_id", "")) == method]
        ax.scatter(
            [_safe_float(row, "unknown_margin_to_detector_threshold_db") for row in method_rows],
            [_safe_float(row, "unknown_component_delta_db") for row in method_rows],
            s=28,
            alpha=0.75,
            color=color,
            label=method,
        )
    ax.axvline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xlabel("unknown BL margin to detected threshold [dB]")
    ax.set_ylabel("unknown component delta [dB]")
    ax.set_title("detected threshold margin vs A2 effective source preservation")
    ax.legend()
    ax.grid(alpha=0.25)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / "detector_margin_vs_unknown_delta.png", dpi=150)
    plt.close(fig)


def _run_source_mask_risk_sweep() -> list[dict[str, object]]:
    """source mask risk sweep を実行し、CSV / report / figures を保存する。

    Returns:
        保存した summary row list。
    """
    scenarios = _risk_scenarios()
    assets = _load_risk_assets(_collect_sweep_frequencies(scenarios))
    rows: list[dict[str, object]] = []
    for scenario_index, scenario in enumerate(scenarios):
        rows.extend(
            _evaluate_risk_scenario(
                scenario=scenario,
                assets=assets,
                seed_offset=scenario_index,
            )
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, SUMMARY_CSV_PATH)
    _write_report(rows)
    _plot_level_snr_heatmap(rows)
    _plot_margin_scatter(rows)
    return rows


def main() -> None:
    """コマンドライン実行 entry point。"""
    rows = _run_source_mask_risk_sweep()
    print(f"wrote {SUMMARY_CSV_PATH} ({len(rows)} rows)")
    print(f"wrote {REPORT_MD_PATH}")
    print(f"wrote {FIGURE_DIR}")


if __name__ == "__main__":
    main()
