"""運用スパースアレイで時間領域 beam-domain SLC の漏れ込み診断を行うモジュール。"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .._validation import require, require_positive_float, require_positive_int
from ..beamforming_evaluation.fractional_response import (
    calculate_fractional_beam_response_matrix,
)
from ..beamforming_evaluation.level_metrics import (
    calculate_real_tone_response_rms_level_db20,
    calculate_rms_level_db20,
)
from ..level_conversion import LevelConverter, level_20log10_rms
from .diagnostic_plotting import require_matplotlib
from .fractional_delay_slc_diagnostics import _run_fractional_delay_diagnostics
from .operational_sparse_array import load_operational_sparse_array
from .slc import BeamDomainSLC, SlcConfig, SlcProcessResult, build_time_tapped_reference_matrix
from .time_delay import FractionalDelayAndSumBeamformer
from .time_delay_diagnostics import TimeDelayDiagnosticConfig, TimeDelayDiagnosticSource

_INPUT_RMS_LEVEL_CONVERTER = LevelConverter.for_definition(
    level_20log10_rms(reference_rms=1.0, reference_label="input RMS")
)


def _calculate_input_rms_level(signal: NDArray[Any]) -> float:
    """共有input RMS契約で波形levelを計算する。"""

    return calculate_rms_level_db20(signal, level_converter=_INPUT_RMS_LEVEL_CONVERTER)


def _finite_float_statistics(values: list[float]) -> dict[str, float | int | None]:
    """有限な scalar 診断値の block 間統計を返す。

    Args:
        values: block ごとの条件数、重みノルム、忘却係数などの scalar 診断値。
            各要素は無次元比であり、配列 shape は持たない。

    Returns:
        `count`, `min`, `max`, `mean`, `std` を持つ辞書。
        有効値がない場合、`count` は 0、その他は `None` とする。

    境界条件:
        NaN / inf は共分散推定の破綻を示すため、統計量から除外して count に反映する。
        呼び出し側は count が block 数より小さい場合を不安定条件として読める。
    """
    finite_values = np.asarray([float(value) for value in values if bool(np.isfinite(float(value)))], dtype=np.float64)
    if finite_values.size == 0:
        # SLC が全 block で無効、または診断量が非有限の場合は統計量を作らない。
        # JSON では None にして、0.0 と誤読されることを避ける。
        return {"count": 0, "min": None, "max": None, "mean": None, "std": None}
    return {
        "count": int(finite_values.size),
        "min": float(np.min(finite_values)),
        "max": float(np.max(finite_values)),
        "mean": float(np.mean(finite_values)),
        "std": float(np.std(finite_values)),
    }


def _covariance_memory_diagnostics(
    *,
    alpha: float | None,
    block_size: int,
    block_time_sec: float,
    memory_time_sec: float,
    enabled_block_count: int,
) -> dict[str, float | int | None]:
    """忘却係数から共分散積分時間の診断量を作る。

    Args:
        alpha: block 間の忘却係数。`None` は SLC が有効化されなかった状態を表す。
        block_size: 1 回の共分散更新に使う sample 数。単位は sample。
        block_time_sec: 1 block の時間長。単位は秒。
        memory_time_sec: 指数忘却の時定数。単位は秒。
        enabled_block_count: SLC 係数更新に成功した block 数。単位は block 数。

    Returns:
        共分散積分時間を評価する scalar 診断値の辞書。

    境界条件:
        `alpha` が 1 に近いほど過去 block を強く残すため、分散は下がるが追従は遅くなる。
        ここでは指数平均の重み列 `(1-alpha) alpha^k` に対し、
        `1 / sum(w_k^2) = (1 + alpha) / (1 - alpha)` を独立 block 数の目安として記録する。
    """
    require_positive_int("block_size", int(block_size))
    require_positive_float("block_time_sec", float(block_time_sec))
    require_positive_float("memory_time_sec", float(memory_time_sec))
    require(int(enabled_block_count) >= 0, "enabled_block_count must be non-negative.")

    if alpha is None:
        return {
            "block_size": int(block_size),
            "block_time_sec": float(block_time_sec),
            "memory_time_sec": float(memory_time_sec),
            "alpha": None,
            "e_folding_block_count": None,
            "asymptotic_effective_independent_block_count": None,
            "asymptotic_effective_independent_sample_count": None,
            "enabled_block_count": int(enabled_block_count),
            "enabled_duration_sec": float(enabled_block_count) * float(block_time_sec),
        }

    alpha_value = float(alpha)
    require(0.0 <= alpha_value < 1.0, "alpha must lie in [0.0, 1.0).")
    e_folding_block_count = float(memory_time_sec) / float(block_time_sec)
    effective_independent_block_count = (1.0 + alpha_value) / max(1.0 - alpha_value, np.finfo(np.float64).eps)
    return {
        "block_size": int(block_size),
        "block_time_sec": float(block_time_sec),
        "memory_time_sec": float(memory_time_sec),
        "alpha": alpha_value,
        "e_folding_block_count": float(e_folding_block_count),
        "asymptotic_effective_independent_block_count": float(effective_independent_block_count),
        "asymptotic_effective_independent_sample_count": float(effective_independent_block_count * float(block_size)),
        "enabled_block_count": int(enabled_block_count),
        "enabled_duration_sec": float(enabled_block_count) * float(block_time_sec),
    }


def _apply_learned_slc_to_component(
    component_beam_output: NDArray[Any],
    target_beam_index: int,
    slc_result: SlcProcessResult,
    eta_override: float | None = None,
) -> NDArray[Any]:
    """mixed 信号で学習した SLC 係数を別成分の beam output へ適用する。

    Args:
        component_beam_output: target-only または interferer-only の固定整相後出力。
            shape は `[n_beam, n_sample]`。axis=0 が beam、axis=1 が時間サンプルである。
        target_beam_index: 評価する target beam index。単位は beam 本数の index。
        slc_result: mixed 信号から得た SLC 結果。`W` と `reference_beams` を含む。
        eta_override: 評価時だけ使う eta。`None` の場合は `slc_result.eta` を使う。
            raw SLC 候補を評価する場合は、safety fallback 前の eta を渡す。

    Returns:
        学習済み SLC 係数を同じ component に適用した target beam 出力。shape は `[n_sample]`。

    Raises:
        ValueError: SLC が無効化されており係数が存在しない場合、または shape が不正な場合。
    """
    beam_output = np.asarray(component_beam_output)
    require(beam_output.ndim == 2, "component_beam_output must have shape (n_beam, n_sample).")
    require(0 <= int(target_beam_index) < beam_output.shape[0], "target_beam_index is out of range.")
    if slc_result.W is None:
        raise ValueError("SLC result does not contain weights because SLC was disabled.")

    reference_beams = np.asarray(slc_result.reference_beams, dtype=np.int64)
    weights = np.asarray(slc_result.W, dtype=np.complex128)
    require(weights.shape[0] == 1, "component evaluation expects one target beam.")
    require(reference_beams.size > 0, "component evaluation requires at least one reference beam.")
    require(
        weights.shape[1] % reference_beams.size == 0,
        "SLC weights must be an integer multiple of reference beam count.",
    )

    target_output = beam_output[int(target_beam_index), :]
    reference_output = beam_output[reference_beams, :]
    if slc_result.reference_blocking_matrix is not None:
        # 学習時に desired response blocking を使った場合、成分別評価でも同じ blocking を適用する。
        # ここを省略すると、raw SLC の成分分解が実際の係数推定時の reference 空間と一致しない。
        reference_output = np.asarray(slc_result.reference_blocking_matrix, dtype=np.complex128) @ reference_output

    tap_len = int(weights.shape[1] // reference_beams.size)
    tapped_reference_output = build_time_tapped_reference_matrix(reference_output=reference_output, tap_len=tap_len)

    # C_valid[n] = Σ_dof conj(w_dof) u_tap[dof, n]。
    # L>1 では先頭 L-1 サンプルに過去 reference が揃わないため、そこは固定整相出力をそのまま残す。
    cancel_estimate = np.conj(weights[0]) @ tapped_reference_output
    effective_eta = float(slc_result.eta if eta_override is None else eta_override)
    component_output = target_output.astype(np.result_type(target_output.dtype, cancel_estimate.dtype), copy=True)
    component_output[tap_len - 1 :] = target_output[tap_len - 1 :] - effective_eta * cancel_estimate
    return component_output


def _process_streaming_slc_blocks(
    beam_output: NDArray[Any],
    target_beam_index: int,
    slc: BeamDomainSLC,
    desired_response_matrix: NDArray[Any],
    slc_analysis_block_size: int,
) -> tuple[NDArray[Any], NDArray[Any], list[tuple[int, int, SlcProcessResult]], SlcProcessResult, float]:
    """固定整相後 beam output を block ごとに SLC 処理する。

    Args:
        beam_output: mixed 条件の固定整相後出力。shape は `[n_beam, n_sample]`。
            axis=0 が beam、axis=1 が時間サンプルである。
        target_beam_index: 評価する target beam index。単位は beam index。
        slc: 忘却統計を保持する `BeamDomainSLC`。
        desired_response_matrix: target 保護用 response。shape は `[n_beam, n_constraint]`。
        slc_analysis_block_size: 共分散を更新する block 長。単位は sample。

    Returns:
        `(Y, C, block_results, last_enabled_result, elapsed_sec)`。
        `Y` と `C` の shape は `[1, n_sample]`、`block_results` は `(start, stop, result)` の列である。

    Raises:
        ValueError: `beam_output` の shape が不正、または全 block で SLC が無効化された場合。

    境界条件:
        共分散は各 block の `R_hat` を `R_k = alpha R_{k-1} + (1-alpha) R_hat_k` として積分する。
        最終 block が短く capacity を満たさない場合は、その block だけ固定整相を通し、統計更新しない。
    """
    beam_signals = np.asarray(beam_output)
    require(beam_signals.ndim == 2, "beam_output must have shape (n_beam, n_sample).")
    require(beam_signals.shape[1] > 0, "beam_output must contain at least one sample.")
    require_positive_int("slc_analysis_block_size", int(slc_analysis_block_size))

    target_indices = np.array([int(target_beam_index)], dtype=np.int64)
    y_blocks: list[NDArray[Any]] = []
    c_blocks: list[NDArray[Any]] = []
    block_results: list[tuple[int, int, SlcProcessResult]] = []
    last_enabled_result: SlcProcessResult | None = None

    process_start_sec = time.perf_counter()
    for block_start in range(0, int(beam_signals.shape[1]), int(slc_analysis_block_size)):
        block_stop = min(block_start + int(slc_analysis_block_size), int(beam_signals.shape[1]))
        # block_signals shape: [n_beam, n_block_sample]。
        # axis=1 の block ごとに共分散を作り、BeamDomainSLC 内部の忘却統計へ順次積分する。
        block_signals = beam_signals[:, block_start:block_stop]
        block_result = slc.process(
            beam_output=block_signals,
            target_beams=target_indices,
            heading_deg=None,
            heading_valid=False,
            desired_response_matrix=desired_response_matrix,
        )
        y_blocks.append(np.asarray(block_result.Y))
        c_blocks.append(np.asarray(block_result.C))
        block_results.append((int(block_start), int(block_stop), block_result))
        if block_result.W is not None:
            last_enabled_result = block_result
    elapsed_sec = time.perf_counter() - process_start_sec

    if last_enabled_result is None:
        # すべての block が capacity 不足で無効化された場合は、SLC の評価値を作れない。
        # 固定整相へ安全に倒すだけでは方式比較にならないため、診断では明示的に停止する。
        raise ValueError("time-domain SLC was disabled for all streaming blocks; reference capacity is insufficient.")

    # Y/C shape: [1, n_sample]。axis=1 を block 順に連結し、後段の成分別 RMS 評価を全時間で行う。
    return (
        np.concatenate(y_blocks, axis=1),
        np.concatenate(c_blocks, axis=1),
        block_results,
        last_enabled_result,
        float(elapsed_sec),
    )


def _apply_streaming_slc_to_component(
    component_beam_output: NDArray[Any],
    target_beam_index: int,
    block_results: list[tuple[int, int, SlcProcessResult]],
    eta_override: float | None = None,
) -> NDArray[Any]:
    """mixed 信号で逐次学習した SLC 係数列を別成分へ block ごとに適用する。

    Args:
        component_beam_output: target-only または interferer-only の固定整相後出力。
            shape は `[n_beam, n_sample]`。axis=0 が beam、axis=1 が時間サンプルである。
        target_beam_index: 評価する target beam index。単位は beam index。
        block_results: mixed 条件から得た `(start, stop, SlcProcessResult)` の列。
        eta_override: 評価時だけ使う eta。`None` の場合は各 block の effective eta を使う。

    Returns:
        block ごとの学習済み SLC 係数を同じ component に適用した target beam 出力。shape は `[n_sample]`。

    境界条件:
        block ごとに係数が異なるため、最後の係数を全区間へ適用しない。
        capacity 不足で無効化された block は、運用と同じく固定整相出力を通す。
    """
    beam_output = np.asarray(component_beam_output)
    require(beam_output.ndim == 2, "component_beam_output must have shape (n_beam, n_sample).")
    require(len(block_results) > 0, "block_results must not be empty.")

    component_blocks: list[NDArray[Any]] = []
    for block_start, block_stop, block_result in block_results:
        component_block = beam_output[:, int(block_start) : int(block_stop)]
        if block_result.W is None:
            # この block は参照容量不足で SLC を更新していないため、同じ区間の成分評価も固定整相を通す。
            component_blocks.append(component_block[int(target_beam_index), :].copy())
        else:
            component_blocks.append(
                _apply_learned_slc_to_component(
                    component_beam_output=component_block,
                    target_beam_index=int(target_beam_index),
                    slc_result=block_result,
                    eta_override=eta_override,
                )
            )
    return np.concatenate(component_blocks, axis=0)


def _resolve_summary_path_or_default(summary: dict[str, object], key: str, default_path: Path) -> str:
    """summary 内の path 文字列を絶対 path へ正規化する。

    Args:
        summary: 診断関数が返す summary。JSON 由来のため値型は `object` として扱う。
        key: path を取り出す key。
        default_path: key がない場合に使う既定 path。

    Returns:
        絶対 path 文字列。

    境界条件:
        summary は JSON 保存対象でもあるため、path は文字列として保存する。
        key が存在しても文字列でない場合は、誤った path を作らず停止する。
    """
    path_value = summary.get(key)
    if path_value is None:
        return str(default_path.resolve())
    if not isinstance(path_value, str):
        raise TypeError(f"summary[{key!r}] must be a string path.")
    return str(Path(path_value).resolve())

def _build_real_signal_desired_response_matrix(
    response_matrix: NDArray[Any],
    target_beam_index: int,
) -> NDArray[np.complex128]:
    """時間領域の実信号 target を保護する desired response 制約を作る。

    Args:
        response_matrix: beam-to-beam の複素応答行列。shape は `[n_beam, n_beam]`。
            axis=0 が観測 beam、axis=1 が到来方向に対応する look beam である。
        target_beam_index: 保護する target beam index。単位は beam index。

    Returns:
        desired response matrix。shape は `[n_beam, 2]`。
        axis=1 の 0 列目は `+f` 側の応答、1 列目は実信号に含まれる `-f` 側の共役応答である。

    Raises:
        ValueError: `response_matrix` が正方 2 次元でない、または index が範囲外の場合。

    境界条件:
        時間領域診断の入力は実数 tone であり、target leakage は `A(f)s_f[n]` と
        `conj(A(f))conj(s_f[n])` の 2 つの複素部分空間を持つ。1 制約だけで blocking すると
        片側の周波数成分が reference に残り、target-only 条件で約 3 dB の自己消去を起こす。
    """
    responses = np.asarray(response_matrix, dtype=np.complex128)
    require(responses.ndim == 2, "response_matrix must have shape (n_beam, n_beam).")
    require(responses.shape[0] == responses.shape[1], "response_matrix must be square.")
    require(0 <= int(target_beam_index) < responses.shape[1], "target_beam_index is out of range.")

    positive_frequency_response = responses[:, int(target_beam_index)]
    # desired_response_matrix shape: [n_beam, 2]。
    # axis=1 は正負周波数の制約であり、実信号 tone の target 成分を reference 空間から両方除去する。
    return np.column_stack((positive_frequency_response, np.conj(positive_frequency_response))).astype(
        np.complex128,
        copy=False,
    )


def _plot_target_leakage_levels(
    output_path: Path,
    levels: dict[str, float],
    title: str,
) -> None:
    """target beam の成分別 before/after RMS レベルを棒グラフで保存する。

    Args:
        output_path: PNG 保存先。
        levels: レベル辞書。値は dB20。
        title: 図タイトル。

    境界条件:
        matplotlib が使えない環境では `require_matplotlib()` が例外を出す。
        診断図が作れない状態で評価を進めると、SLC 前後の誤判定を見逃すため停止する。
    """
    require_matplotlib()
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = list(levels.keys())
    values = np.array([float(levels[label]) for label in labels], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    ax.bar(np.arange(len(labels)), values, color=["#2b6cb0", "#2f855a", "#c53030", "#dd6b20"][: len(labels)])
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=20.0, ha="right")
    ax.set_ylabel("RMS level [dB re input RMS]")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)




@dataclass(frozen=True)
class _ProtectedTargetSlcBlLevels:
    """保護 target beam を固定した SLC 前後 BL レベルを保持する。

    この内部クラスは、SLC の target beam を評価対象方位に固定し、その target 出力に対する
    source 方位走査応答を保持する。各配列の shape は `[n_look]` で、axis=0 は source 方位を表す。

    単位は `dB re input RMS` であり、before/after 差は `dB re before level` として読む。
    """

    target_before_db20: NDArray[np.float64]
    target_after_db20: NDArray[np.float64]
    interferer_before_db20: NDArray[np.float64]
    interferer_after_db20: NDArray[np.float64]


def _calculate_protected_target_slc_response_curve(
    *,
    response_matrix: NDArray[Any],
    target_beam_index: int,
    slc_result: SlcProcessResult,
    slc_config: SlcConfig,
    frequency_hz: float,
    fs_hz: float,
    source_level_db20: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """保護 target beam の SLC 前後空間応答を source 方位ごとに計算する。

    Args:
        response_matrix: 固定整相の beam-to-look 応答。shape は `[n_beam, n_look]`。
            axis=0 が観測 beam、axis=1 が source/look 方位である。
        target_beam_index: SLC で保護する target beam index。単位は beam index。
        slc_result: mixed 条件で保護 target beam に対して学習した SLC 結果。
        slc_config: SLC 設定。tap_len と eta を読む。
        frequency_hz: 評価する source 周波数。単位は Hz。
        fs_hz: サンプリング周波数。単位は Hz。
        source_level_db20: source 入力 RMS レベル。単位は dB re input RMS。

    Returns:
        `(before_db20, after_db20)`。
        どちらも shape は `[n_look]`、単位は dB re input RMS。

    Raises:
        ValueError: SLC 係数が存在しない、または shape が不正な場合。

    境界条件:
        SLC は全 beam 出力を別々に作るビームフォーマではなく、保護 target beam の後段キャンセラである。
        そのため BL は `target_beam_index` の出力を固定し、source 方位を走査した応答として定義する。
    """
    responses = np.asarray(response_matrix, dtype=np.complex128)
    require(responses.ndim == 2, "response_matrix must have shape (n_beam, n_look).")
    require(0 <= int(target_beam_index) < responses.shape[0], "target_beam_index is out of range.")
    require_positive_float("frequency_hz", float(frequency_hz))
    require_positive_float("fs_hz", float(fs_hz))
    if slc_result.W is None:
        raise ValueError("SLC result does not contain weights because SLC was disabled.")

    reference_beams = np.asarray(slc_result.reference_beams, dtype=np.int64)
    weights = np.asarray(slc_result.W, dtype=np.complex128)
    require(weights.shape[0] == 1, "protected target BL expects one protected target beam.")
    require(reference_beams.size > 0, "protected target BL requires at least one reference beam.")
    require(weights.shape[1] % reference_beams.size == 0, "SLC weights must be an integer multiple of reference beam count.")

    # fixed_response shape: [n_look]。target beam 出力だけを固定して source 方位応答を見る。
    # 実 FIR の固定整相では `H(-f)=conj(H(+f))` だが、後段の複素 SLC 係数ではこの対称性が崩れ得る。
    fixed_positive_response = responses[int(target_beam_index), :]
    fixed_negative_response = np.conj(fixed_positive_response)
    reference_positive_response = responses[reference_beams, :]
    reference_negative_response = np.conj(reference_positive_response)
    if slc_result.reference_blocking_matrix is not None:
        blocking_matrix = np.asarray(slc_result.reference_blocking_matrix, dtype=np.complex128)
        # 学習時と同じ desired blocking を BL 評価にも適用しないと、評価上だけ target 成分が reference に残る。
        # blocking matrix は時間領域の reference 信号そのものへ掛けるため、-f 側にも同じ B を掛ける。
        reference_positive_response = blocking_matrix @ reference_positive_response
        reference_negative_response = blocking_matrix @ reference_negative_response

    tap_len = int(weights.shape[1] // reference_beams.size)
    require(tap_len == int(slc_config.tap_len), "SLC weight tap length and config.tap_len must agree.")

    tapped_positive_responses: list[NDArray[np.complex128]] = []
    tapped_negative_responses: list[NDArray[np.complex128]] = []
    for lag_index in range(tap_len):
        # build_time_tapped_reference_matrix と同じ並びで、lag=0 は現在サンプル、lag>0 は過去サンプルを表す。
        # +f 側の過去サンプルは exp(-jωlag)、-f 側は exp(+jωlag) の位相になる。
        positive_lag_phase = np.exp(-1j * 2.0 * np.pi * float(frequency_hz) * float(lag_index) / float(fs_hz))
        negative_lag_phase = np.exp(1j * 2.0 * np.pi * float(frequency_hz) * float(lag_index) / float(fs_hz))
        tapped_positive_responses.append(np.asarray(reference_positive_response * positive_lag_phase, dtype=np.complex128))
        tapped_negative_responses.append(np.asarray(reference_negative_response * negative_lag_phase, dtype=np.complex128))
    tapped_positive_reference = np.concatenate(tapped_positive_responses, axis=0)
    tapped_negative_reference = np.concatenate(tapped_negative_responses, axis=0)

    positive_cancel_response = np.conj(weights[0]) @ tapped_positive_reference
    negative_cancel_response = np.conj(weights[0]) @ tapped_negative_reference
    effective_eta = float(slc_result.eta)
    positive_after_response = fixed_positive_response - effective_eta * positive_cancel_response
    negative_after_response = fixed_negative_response - effective_eta * negative_cancel_response
    source_rms = _INPUT_RMS_LEVEL_CONVERTER.input_to_rms(float(source_level_db20))

    before_db20 = calculate_real_tone_response_rms_level_db20(
        fixed_positive_response,
        fixed_negative_response,
        source_rms,
        level_converter=_INPUT_RMS_LEVEL_CONVERTER,
    )
    after_db20 = calculate_real_tone_response_rms_level_db20(
        positive_after_response,
        negative_after_response,
        source_rms,
        level_converter=_INPUT_RMS_LEVEL_CONVERTER,
    )
    return np.asarray(before_db20, dtype=np.float64), np.asarray(after_db20, dtype=np.float64)


def _calculate_protected_target_slc_bl_levels(
    *,
    target_response_matrix: NDArray[Any],
    interferer_response_matrix: NDArray[Any],
    target_beam_index: int,
    slc_result: SlcProcessResult,
    slc_config: SlcConfig,
    fs_hz: float,
    target_frequency_hz: float,
    interferer_frequency_hz: float,
    target_level_db20: float,
    interferer_level_db20: float,
) -> _ProtectedTargetSlcBlLevels:
    """target / interferer 周波数で保護 target beam の SLC 前後 BL を計算する。

    Args:
        target_response_matrix: target 周波数の固定整相応答。shape は `[n_beam, n_look]`。
        interferer_response_matrix: interferer 周波数の固定整相応答。shape は `[n_beam, n_look]`。
        target_beam_index: 保護 target beam index。単位は beam index。
        slc_result: mixed 条件で保護 target beam に対して学習した SLC 結果。
        slc_config: SLC 設定。
        fs_hz: サンプリング周波数。単位は Hz。
        target_frequency_hz: target 周波数。単位は Hz。
        interferer_frequency_hz: interferer 周波数。単位は Hz。
        target_level_db20: target 入力 RMS レベル。単位は dB re input RMS。
        interferer_level_db20: interferer 入力 RMS レベル。単位は dB re input RMS。

    Returns:
        保護 target beam に対する target 周波数応答と interferer 周波数応答の before/after BL。
    """
    target_before, target_after = _calculate_protected_target_slc_response_curve(
        response_matrix=target_response_matrix,
        target_beam_index=int(target_beam_index),
        slc_result=slc_result,
        slc_config=slc_config,
        frequency_hz=float(target_frequency_hz),
        fs_hz=float(fs_hz),
        source_level_db20=float(target_level_db20),
    )
    interferer_before, interferer_after = _calculate_protected_target_slc_response_curve(
        response_matrix=interferer_response_matrix,
        target_beam_index=int(target_beam_index),
        slc_result=slc_result,
        slc_config=slc_config,
        frequency_hz=float(interferer_frequency_hz),
        fs_hz=float(fs_hz),
        source_level_db20=float(interferer_level_db20),
    )
    return _ProtectedTargetSlcBlLevels(
        target_before_db20=target_before,
        target_after_db20=target_after,
        interferer_before_db20=interferer_before,
        interferer_after_db20=interferer_after,
    )


def _find_first_sidelobe_peak_index(levels_db20: NDArray[Any], guard_start: int, guard_stop: int) -> int:
    """guard 外で mainlobe に最も近い第一副極 peak index を返す。

    Args:
        levels_db20: BL レベル。shape は `[n_look]`、単位は dB re input RMS。
        guard_start: mainlobe guard の開始 index。単位は look index。
        guard_stop: mainlobe guard の終端 index。Python slice と同じく exclusive である。

    Returns:
        第一副極として扱う peak index。左右の第一局所 peak のうち、レベルが高い側を返す。

    境界条件:
        方位 grid が粗い場合、guard 外に明確な局所最大が存在しないことがある。
        その場合は、その側の guard 外最大点へ fallback し、評価項目を欠落させない。
    """
    levels = np.asarray(levels_db20, dtype=np.float64)
    require(levels.ndim == 1, "levels_db20 must have shape (n_look,).")
    require(0 <= int(guard_start) < int(guard_stop) <= levels.size, "guard range is out of bounds.")

    def first_local_peak(search_indices: list[int]) -> int | None:
        """guard 境界から外側へ見て最初の局所最大を返す。"""
        for candidate_index in search_indices:
            left_is_lower = candidate_index == 0 or levels[candidate_index] >= levels[candidate_index - 1]
            right_is_lower = candidate_index == levels.size - 1 or levels[candidate_index] >= levels[candidate_index + 1]
            if bool(left_is_lower and right_is_lower):
                return int(candidate_index)
        if len(search_indices) == 0:
            return None

        # 明確な局所最大が無い単調区間では、第一副極を定義できない。
        # 評価を落とさないため、その側の guard 外最大点を conservative fallback として使う。
        side_levels = levels[np.asarray(search_indices, dtype=np.int64)]
        return int(search_indices[int(np.argmax(side_levels))])

    left_search_indices = list(range(int(guard_start) - 1, -1, -1))
    right_search_indices = list(range(int(guard_stop), int(levels.size)))
    first_peak_candidates: list[int] = []
    left_peak = first_local_peak(left_search_indices)
    right_peak = first_local_peak(right_search_indices)
    if left_peak is not None:
        first_peak_candidates.append(int(left_peak))
    if right_peak is not None:
        first_peak_candidates.append(int(right_peak))
    require(len(first_peak_candidates) > 0, "first sidelobe requires at least one guard outside sample.")

    candidate_levels = levels[np.asarray(first_peak_candidates, dtype=np.int64)]
    return int(first_peak_candidates[int(np.argmax(candidate_levels))])

def _protected_target_bl_sidelobe_metrics(
    *,
    axis_az_deg: NDArray[Any],
    before_levels_db20: NDArray[Any],
    after_levels_db20: NDArray[Any],
    target_beam_index: int,
    marker_azimuth_deg: float,
    guard_beam_count: int,
) -> dict[str, float | int]:
    """保護 target beam 固定 BL の guard 外サイドローブ指標を計算する。

    Args:
        axis_az_deg: source 方位軸。shape は `[n_look]`、単位は deg。
        before_levels_db20: SLC 前の保護 target beam 応答。shape は `[n_look]`。
        after_levels_db20: SLC 後の保護 target beam 応答。shape は `[n_look]`。
        target_beam_index: 保護 target beam index。単位は beam index。
        marker_azimuth_deg: 評価したい source 方位。単位は deg。
        guard_beam_count: target 周辺を mainlobe として除外する beam 本数。

    Returns:
        guard 外最大レベル、guard 外最大悪化量、marker 方位での低下量を含む辞書。
        レベルは `dB re input RMS`、差分は `dB re before level` として読む。

    境界条件:
        SLC は局所的な干渉キャンセルであり、guard 外全体の sidelobe floor を必ず下げるとは限らない。
        そのため、marker 方位の低下と guard 外 peak / 最大悪化を分けて評価する。
    """
    azimuths = np.asarray(axis_az_deg, dtype=np.float64)
    before_levels = np.asarray(before_levels_db20, dtype=np.float64)
    after_levels = np.asarray(after_levels_db20, dtype=np.float64)
    require(azimuths.ndim == 1, "axis_az_deg must have shape (n_look,).")
    require(before_levels.shape == azimuths.shape, "before_levels_db20 must match axis_az_deg shape.")
    require(after_levels.shape == azimuths.shape, "after_levels_db20 must match axis_az_deg shape.")
    require(0 <= int(target_beam_index) < azimuths.size, "target_beam_index is out of range.")
    require(int(guard_beam_count) >= 0, "guard_beam_count must be non-negative.")

    n_look = int(azimuths.size)
    guard_start = max(0, int(target_beam_index) - int(guard_beam_count))
    guard_stop = min(n_look, int(target_beam_index) + int(guard_beam_count) + 1)
    outside_guard_mask = np.ones(n_look, dtype=np.bool_)
    outside_guard_mask[guard_start:guard_stop] = False
    require(bool(np.any(outside_guard_mask)), "guard outside region must contain at least one look direction.")

    before_outside = before_levels[outside_guard_mask]
    after_outside = after_levels[outside_guard_mask]
    delta_outside = after_outside - before_outside
    outside_indices = np.flatnonzero(outside_guard_mask)
    before_peak_local_index = int(np.argmax(before_outside))
    after_peak_local_index = int(np.argmax(after_outside))
    max_worsening_local_index = int(np.argmax(delta_outside))
    marker_index = int(np.argmin(np.abs(azimuths - float(marker_azimuth_deg))))
    before_first_sidelobe_index = _find_first_sidelobe_peak_index(before_levels, guard_start=guard_start, guard_stop=guard_stop)
    after_first_sidelobe_index = _find_first_sidelobe_peak_index(after_levels, guard_start=guard_start, guard_stop=guard_stop)

    return {
        "guard_beam_count": int(guard_beam_count),
        "guard_start_index": int(guard_start),
        "guard_stop_index_exclusive": int(guard_stop),
        "before_guard_outside_peak_db20": float(before_outside[before_peak_local_index]),
        "before_guard_outside_peak_azimuth_deg": float(azimuths[int(outside_indices[before_peak_local_index])]),
        "after_guard_outside_peak_db20": float(after_outside[after_peak_local_index]),
        "after_guard_outside_peak_azimuth_deg": float(azimuths[int(outside_indices[after_peak_local_index])]),
        "guard_outside_peak_delta_db": float(after_outside[after_peak_local_index] - before_outside[before_peak_local_index]),
        "before_first_sidelobe_peak_db20": float(before_levels[before_first_sidelobe_index]),
        "before_first_sidelobe_peak_azimuth_deg": float(azimuths[before_first_sidelobe_index]),
        "after_first_sidelobe_peak_db20": float(after_levels[after_first_sidelobe_index]),
        "after_first_sidelobe_peak_azimuth_deg": float(azimuths[after_first_sidelobe_index]),
        "first_sidelobe_peak_delta_db": float(
            after_levels[after_first_sidelobe_index] - before_levels[before_first_sidelobe_index]
        ),
        "first_sidelobe_reduction_db": float(
            before_levels[before_first_sidelobe_index] - after_levels[after_first_sidelobe_index]
        ),
        "max_guard_outside_worsening_db": float(delta_outside[max_worsening_local_index]),
        "max_guard_outside_worsening_azimuth_deg": float(azimuths[int(outside_indices[max_worsening_local_index])]),
        "marker_azimuth_deg": float(azimuths[marker_index]),
        "before_at_marker_db20": float(before_levels[marker_index]),
        "after_at_marker_db20": float(after_levels[marker_index]),
        "reduction_at_marker_db": float(before_levels[marker_index] - after_levels[marker_index]),
    }

def _plot_slc_bl_overlay(
    *,
    output_path: Path,
    axis_az_deg: NDArray[Any],
    before_levels_db20: NDArray[Any],
    after_levels_db20: NDArray[Any],
    marker_azimuth_deg: float,
    marker_label: str,
    title: str,
    caption: str,
    before_label: str,
    after_label: str,
) -> None:
    """固定整相後 BL と SLC 後 BL を重ね書きする。

    Args:
        output_path: PNG 保存先。
        axis_az_deg: 方位軸。shape は `[n_look]`、単位は deg。
        before_levels_db20: 固定整相後の保護 target beam 応答。shape は `[n_look]`。
        after_levels_db20: SLC 後の保護 target beam 応答。shape は `[n_look]`。
        marker_azimuth_deg: 注目する source 方位。単位は deg。
        marker_label: 注目方位の凡例名。
        title: 図タイトル。
        caption: 図下部の説明。
        before_label: SLC 前の凡例名。
        after_label: SLC 後の凡例名。

    境界条件:
        ここでの BL は出力 beam 方位ではなく、保護 target beam を固定した source 方位応答である。
        peak 方位は連続角ではなく、離散 look grid 上の最大値として表示する。
        marker の数値は最近傍 grid 点で読む。真方位 exact response ではないため、
        図上で local leakage を判断するときは summary JSON の marker 方位も併記する。
    """
    require_matplotlib()
    import matplotlib.pyplot as plt

    azimuths = np.asarray(axis_az_deg, dtype=np.float64)
    before_levels = np.asarray(before_levels_db20, dtype=np.float64)
    after_levels = np.asarray(after_levels_db20, dtype=np.float64)
    require(azimuths.ndim == 1, "axis_az_deg must have shape (n_look,).")
    require(before_levels.shape == azimuths.shape, "before_levels_db20 must match axis_az_deg shape.")
    require(after_levels.shape == azimuths.shape, "after_levels_db20 must match axis_az_deg shape.")

    before_peak_index = int(np.argmax(before_levels))
    after_peak_index = int(np.argmax(after_levels))
    marker_index = int(np.argmin(np.abs(azimuths - float(marker_azimuth_deg))))
    marker_before_db20 = float(before_levels[marker_index])
    marker_after_db20 = float(after_levels[marker_index])
    marker_reduction_db = float(marker_before_db20 - marker_after_db20)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    ax.plot(azimuths, before_levels, linewidth=1.5, color="tab:blue", label=before_label)
    ax.plot(azimuths, after_levels, linewidth=1.5, color="tab:orange", label=after_label)
    ax.axvline(float(marker_azimuth_deg), color="black", linestyle=":", linewidth=1.0, label=marker_label)
    ax.axvline(float(azimuths[before_peak_index]), color="tab:blue", linestyle="--", linewidth=1.0, label="Before peak")
    ax.axvline(float(azimuths[after_peak_index]), color="tab:orange", linestyle="-.", linewidth=1.0, label="After peak")
    ax.set_xlabel("Source azimuth [deg]")
    ax.set_ylabel("RMS Level [dB re input RMS]")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    # 図だけを見たときに「低減」か「悪化」かを誤読しないよう、
    # marker 最近傍 grid 点の before/after と before-after 差を明示する。
    annotation = (
        f"marker grid: {azimuths[marker_index]:.2f} deg\n"
        f"before: {marker_before_db20:.2f} dB re input RMS\n"
        f"after: {marker_after_db20:.2f} dB re input RMS\n"
        f"before-after: {marker_reduction_db:+.2f} dB"
    )
    ax.text(
        0.02,
        0.98,
        annotation,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "0.7", "alpha": 0.9},
    )
    fig.text(0.03, 0.01, caption, ha="left", va="bottom", fontsize=9)
    fig.tight_layout(rect=(0.03, 0.05, 1.0, 0.96))
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

def _plot_slc_waveform_overlay(
    output_path: Path,
    *,
    fixed_mixed_target: NDArray[Any],
    raw_mixed_target: NDArray[Any],
    effective_mixed_target: NDArray[Any],
    fs_hz: float,
    display_duration_sec: float = 0.01,
) -> None:
    """SLC 前後の target beam 時系列を同じ軸へ重ねて保存する。

    Args:
        output_path: PNG 保存先。
        fixed_mixed_target: SLC 前の mixed target beam 出力。shape は `[n_sample]`。
        raw_mixed_target: safety gate 前の SLC 出力。shape は `[n_sample]`。
        effective_mixed_target: safety gate 後の運用出力。shape は `[n_sample]`。
        fs_hz: サンプリング周波数。単位は Hz。
        display_duration_sec: 図に表示する先頭時間。単位は秒。

    境界条件:
        SLC 出力は複素最小二乗係数により複素値になり得る。
        時系列表示では物理入力と同じ実部を重ね、虚部はスペクトル評価で確認する。
    """
    require_matplotlib()
    import matplotlib.pyplot as plt

    require_positive_float("fs_hz", float(fs_hz))
    require_positive_float("display_duration_sec", float(display_duration_sec))

    before = np.asarray(fixed_mixed_target)
    raw_after = np.asarray(raw_mixed_target)
    effective_after = np.asarray(effective_mixed_target)
    require(before.ndim == 1, "fixed_mixed_target must have shape (n_sample,).")
    require(raw_after.ndim == 1, "raw_mixed_target must have shape (n_sample,).")
    require(effective_after.ndim == 1, "effective_mixed_target must have shape (n_sample,).")
    require(before.shape[0] == raw_after.shape[0] == effective_after.shape[0], "overlay signals must share n_sample.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_display_sample = min(int(before.shape[0]), max(1, int(round(float(display_duration_sec) * float(fs_hz)))))
    # time_axis_sec shape: [n_display_sample]。axis=0 は時間サンプルで、単位は秒。
    time_axis_sec = np.arange(n_display_sample, dtype=np.float64) / float(fs_hz)

    fig, ax = plt.subplots(figsize=(11.0, 4.8))
    ax.plot(time_axis_sec, np.real(before[:n_display_sample]), label="before fixed BF", linewidth=1.4)
    ax.plot(time_axis_sec, np.real(raw_after[:n_display_sample]), label="after raw SLC", linewidth=1.1, alpha=0.85)
    ax.plot(
        time_axis_sec,
        np.real(effective_after[:n_display_sample]),
        label="after effective",
        linewidth=1.0,
        alpha=0.75,
        linestyle="--",
    )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude [re input RMS]")
    ax.set_title("Target-beam waveform overlay before/after time-domain SLC")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _amplitude_spectrum_db20(signal: NDArray[Any], fs_hz: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """信号の振幅スペクトルを dB20 で返す。

    Args:
        signal: 評価信号。shape は `[n_sample]`。
        fs_hz: サンプリング周波数。単位は Hz。

    Returns:
        `(freq_hz, level_db20)`。
        `freq_hz` と `level_db20` の shape は `[n_sample]` で、FFT shift 後の周波数軸である。

    境界条件:
        SLC 後信号は複素になり得るため、実信号専用の rFFT ではなく通常の FFT を使う。
        振幅は `FFT / N` として正規化し、dB 表示は `dB re input RMS` として読む。
    """
    require_positive_float("fs_hz", float(fs_hz))
    values = np.asarray(signal)
    require(values.ndim == 1, "signal must have shape (n_sample,).")
    require(values.shape[0] > 0, "signal must contain at least one sample.")

    n_sample = int(values.shape[0])
    # FFT axis=0 は時間サンプル軸である。fftshift により負周波数から正周波数へ並べる。
    spectrum = np.fft.fftshift(np.fft.fft(values, axis=0)) / float(n_sample)
    freq_hz = np.fft.fftshift(np.fft.fftfreq(n_sample, d=1.0 / float(fs_hz)))
    level_db20 = 20.0 * np.log10(np.maximum(np.abs(spectrum), np.finfo(np.float64).tiny))
    return np.asarray(freq_hz, dtype=np.float64), np.asarray(level_db20, dtype=np.float64)


def _plot_slc_component_spectrum_overlay(
    output_path: Path,
    *,
    fixed_target_component: NDArray[Any],
    raw_target_component: NDArray[Any],
    fixed_interferer_component: NDArray[Any],
    raw_interferer_component: NDArray[Any],
    fs_hz: float,
    target_frequency_hz: float,
    interferer_frequency_hz: float,
    display_half_width_hz: float = 3000.0,
) -> None:
    """target-only / interferer-only 成分の SLC 前後スペクトルを重ねて保存する。

    Args:
        output_path: PNG 保存先。
        fixed_target_component: SLC 前の target-only target beam 出力。shape は `[n_sample]`。
        raw_target_component: SLC 後の target-only target beam 出力。shape は `[n_sample]`。
        fixed_interferer_component: SLC 前の interferer-only target beam 出力。shape は `[n_sample]`。
        raw_interferer_component: SLC 後の interferer-only target beam 出力。shape は `[n_sample]`。
        fs_hz: サンプリング周波数。単位は Hz。
        target_frequency_hz: target tone 周波数。単位は Hz。
        interferer_frequency_hz: interferer tone 周波数。単位は Hz。
        display_half_width_hz: 表示する周波数範囲の半幅。単位は Hz。

    境界条件:
        SLC の性能は mixed 総量だけでは判断しない。
        target-only と interferer-only を分けて重ね、target 保護と interferer 低減を同時に確認する。
    """
    require_matplotlib()
    import matplotlib.pyplot as plt

    require_positive_float("fs_hz", float(fs_hz))
    require_positive_float("target_frequency_hz", float(target_frequency_hz))
    require_positive_float("interferer_frequency_hz", float(interferer_frequency_hz))
    require_positive_float("display_half_width_hz", float(display_half_width_hz))

    target_freq_hz, target_before_db = _amplitude_spectrum_db20(fixed_target_component, fs_hz=float(fs_hz))
    _, target_after_db = _amplitude_spectrum_db20(raw_target_component, fs_hz=float(fs_hz))
    interferer_freq_hz, interferer_before_db = _amplitude_spectrum_db20(fixed_interferer_component, fs_hz=float(fs_hz))
    _, interferer_after_db = _amplitude_spectrum_db20(raw_interferer_component, fs_hz=float(fs_hz))

    center_frequency_hz = 0.5 * (float(target_frequency_hz) + float(interferer_frequency_hz))
    lower_hz = max(0.0, center_frequency_hz - float(display_half_width_hz))
    upper_hz = min(0.5 * float(fs_hz), center_frequency_hz + float(display_half_width_hz))
    # mask shape: [n_bin]。正周波数側だけを表示し、target/interferer tone の近傍を同じ軸で比較する。
    positive_band_mask = (target_freq_hz >= lower_hz) & (target_freq_hz <= upper_hz)
    require(bool(np.any(positive_band_mask)), "spectrum display band must contain at least one FFT bin.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11.0, 5.2))
    ax.plot(target_freq_hz[positive_band_mask], target_before_db[positive_band_mask], label="target before", linewidth=1.2)
    ax.plot(target_freq_hz[positive_band_mask], target_after_db[positive_band_mask], label="target after SLC", linewidth=1.2)
    ax.plot(
        interferer_freq_hz[positive_band_mask],
        interferer_before_db[positive_band_mask],
        label="interferer before",
        linewidth=1.2,
    )
    ax.plot(
        interferer_freq_hz[positive_band_mask],
        interferer_after_db[positive_band_mask],
        label="interferer after SLC",
        linewidth=1.2,
    )
    ax.axvline(float(target_frequency_hz), color="#2b6cb0", linestyle=":", linewidth=1.0)
    ax.axvline(float(interferer_frequency_hz), color="#c53030", linestyle=":", linewidth=1.0)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Amplitude spectrum [dB re input RMS]")
    ax.set_title("Target-beam component spectrum overlay before/after time-domain SLC")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


@dataclass(frozen=True)
class OperationalTimeDomainSlcDiagnosticConfig:
    """運用アレイの時間領域 SLC 漏れ込み診断条件を保持する。

    このクラスは、運用スパースアレイ JSON、保存済み小数遅延 FIR バンク、
    評価周波数、target / interferer 方位、待受ビーム数、出力先を保持する。

    入力はファイルパスと音源条件であり、出力は
    `run_operational_time_domain_slc_leakage_diagnostics()` が保存する
    target-centric SLC 漏れ込み summary と診断図である。

    小数遅延 FIR の設計、アレイ配置の探索、SLC 係数更新方式そのものの変更は責務に含めない。
    信号処理上は、固定整相後 beam_output を block ごとに処理し、忘却係数付きで時間領域共分散を積分する SLC が
    target beam 上の interferer leakage を下げるかを判定する診断条件に位置づく。
    """

    output_dir: Path
    operational_array_definition_path: Path
    fractional_delay_filter_bank_path: Path
    processing_frequency_hz: float = 10000.0
    interferer_frequency_hz: float | None = None
    target_azimuth_deg: float = 90.0
    interferer_azimuth_deg: float = 60.0
    target_level_db20: float = 0.0
    interferer_level_db20: float = -6.0
    duration_s: float = 1.0
    n_beam_az_real: int = 151
    slc_analysis_block_size: int = 8192
    noise_level_db20: float = -300.0
    random_seed: int = 1234
    target_amplitude_modulation_hz: float = 0.7
    target_amplitude_modulation_depth: float = 0.6
    interferer_amplitude_modulation_hz: float = 1.3
    interferer_amplitude_modulation_depth: float = 0.8
    interferer_amplitude_modulation_phase_deg: float = 70.0

    def __post_init__(self) -> None:
        """診断条件の単位、範囲、入力ファイル存在を検証する。"""
        require(Path(self.operational_array_definition_path).exists(), "operational_array_definition_path must exist.")
        require(Path(self.fractional_delay_filter_bank_path).exists(), "fractional_delay_filter_bank_path must exist.")
        require_positive_float("processing_frequency_hz", float(self.processing_frequency_hz))
        if self.interferer_frequency_hz is not None:
            require_positive_float("interferer_frequency_hz", float(self.interferer_frequency_hz))
        require_positive_float("duration_s", float(self.duration_s))
        require_positive_int("n_beam_az_real", int(self.n_beam_az_real))
        require_positive_int("slc_analysis_block_size", int(self.slc_analysis_block_size))
        require(0.0 <= float(self.target_azimuth_deg) <= 180.0, "target_azimuth_deg must lie in [0, 180].")
        require(0.0 <= float(self.interferer_azimuth_deg) <= 180.0, "interferer_azimuth_deg must lie in [0, 180].")
        require(0.0 <= float(self.target_amplitude_modulation_depth) <= 1.0, "target AM depth must lie in [0, 1].")
        require(0.0 <= float(self.interferer_amplitude_modulation_depth) <= 1.0, "interferer AM depth must lie in [0, 1].")


def _build_source_specs(
    config: OperationalTimeDomainSlcDiagnosticConfig,
    *,
    include_target: bool,
    include_interferer: bool,
) -> tuple[TimeDelayDiagnosticSource, ...]:
    """診断ケース用の source spec を作る。

    Args:
        config: 診断条件。
        include_target: target source を含める場合は `True`。
        include_interferer: interferer source を含める場合は `True`。

    Returns:
        `TimeDelayDiagnosticConfig` に渡す source spec。shape は source 数分の tuple。

    Raises:
        ValueError: target も interferer も含めない場合。
    """
    source_specs: list[TimeDelayDiagnosticSource] = []
    if include_target:
        source_specs.append(
            TimeDelayDiagnosticSource(
                azimuth_deg=float(config.target_azimuth_deg),
                frequency_hz=float(config.processing_frequency_hz),
                level_db20=float(config.target_level_db20),
                amplitude_modulation_hz=float(config.target_amplitude_modulation_hz),
                amplitude_modulation_depth=float(config.target_amplitude_modulation_depth),
                label="target",
            )
        )
    if include_interferer:
        source_specs.append(
            TimeDelayDiagnosticSource(
                azimuth_deg=float(config.interferer_azimuth_deg),
                frequency_hz=float(config.processing_frequency_hz if config.interferer_frequency_hz is None else config.interferer_frequency_hz),
                level_db20=float(config.interferer_level_db20),
                amplitude_modulation_hz=float(config.interferer_amplitude_modulation_hz),
                amplitude_modulation_depth=float(config.interferer_amplitude_modulation_depth),
                amplitude_modulation_phase_deg=float(config.interferer_amplitude_modulation_phase_deg),
                label="interferer",
            )
        )
    if not source_specs:
        raise ValueError("at least one source must be enabled.")
    return tuple(source_specs)


def _run_fixed_fractional_case(
    config: OperationalTimeDomainSlcDiagnosticConfig,
    case_name: str,
    active_positions_m: NDArray[np.float64],
    source_specs: tuple[TimeDelayDiagnosticSource, ...],
    fs_hz: float,
    sound_speed_m_s: float,
) -> tuple[dict[str, object], FractionalDelayAndSumBeamformer, NDArray[np.float64], NDArray[np.float64]]:
    """1 ケースの小数遅延固定整相を実行し、summary と beam output を返す。

    Args:
        config: 診断条件。
        case_name: 出力ディレクトリ名。
        active_positions_m: 評価周波数で使う active センサ座標。shape は `[n_active_ch, 3]`、単位は m。
        source_specs: 音源条件。
        fs_hz: サンプリング周波数。単位は Hz。
        sound_speed_m_s: 音速。単位は m/s。

    Returns:
        `(summary, beamformer, beam_output, axis_az_deg)`。
        `beam_output` の shape は `[n_beam, n_sample]`、`axis_az_deg` の shape は `[n_beam]`。
        `beamformer` は小数遅延 FIR 応答込みの beam response matrix を作るために返す。
    """
    diagnostic_config = TimeDelayDiagnosticConfig(
        output_dir=Path(config.output_dir) / case_name,
        fs_hz=float(fs_hz),
        duration_s=float(config.duration_s),
        sound_speed_m_s=float(sound_speed_m_s),
        source_specs=source_specs,
        noise_level_db20=float(config.noise_level_db20),
        random_seed=int(config.random_seed),
        array_n_ch=int(active_positions_m.shape[0]),
        array_positions_m=np.asarray(active_positions_m, dtype=np.float64),
        n_beam_az_real=int(config.n_beam_az_real),
        n_beam_az_virtual=0,
        btr_block_size=1024,
    )

    summary, beamformer, beam_output, axis_az_deg, _, _ = _run_fractional_delay_diagnostics(
        config=diagnostic_config,
        fractional_delay_filter_bank_path=Path(config.fractional_delay_filter_bank_path),
    )
    if not isinstance(beamformer, FractionalDelayAndSumBeamformer):
        # 小数遅延応答行列を作るには delay_table と FIR 応答を持つ整相器が必要である。
        # 型が合わない場合は設計前提が崩れているため、object のまま後段へ渡さずここで停止する。
        raise TypeError("fractional diagnostics must return FractionalDelayAndSumBeamformer.")
    return summary, beamformer, np.asarray(beam_output, dtype=np.float64), np.asarray(axis_az_deg, dtype=np.float64)


def run_operational_time_domain_slc_leakage_diagnostics(
    config: OperationalTimeDomainSlcDiagnosticConfig,
    slc_config: SlcConfig,
) -> dict[str, object]:
    """運用アレイで時間領域 SLC の target beam 漏れ込みを評価する。

    Args:
        config: 運用アレイ、評価周波数、target / interferer 条件、出力先。
        slc_config: 時間領域 SLC 設定。`guard`、`loading`、`min_ref` などを含む。

    Returns:
        target beam 上の mixed / target / interferer 成分レベル、SLC 前後差、
        safety 判定、保存先を含む summary。

    Raises:
        ValueError: 入力 shape が不正、または SLC が参照不足で無効化された場合。
    """
    require_matplotlib()

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    array_definition = load_operational_sparse_array(Path(config.operational_array_definition_path))
    active_indices = array_definition.active_channel_indices_for_frequency(float(config.processing_frequency_hz))

    # positions_m は物理全 CH の shape `[n_ch, 3]` である。
    # 周波数ごとの active subset だけで固定整相を行い、高域で設計外の外側疎配置を混ぜない。
    active_positions_m = np.asarray(array_definition.positions_m[active_indices], dtype=np.float64)
    active_aperture_m = float(np.max(active_positions_m[:, 0]) - np.min(active_positions_m[:, 0]))

    fixed_mixed_summary, mixed_beamformer, mixed_beam_output, axis_az_deg = _run_fixed_fractional_case(
        config=config,
        case_name="fixed_mixed",
        active_positions_m=active_positions_m,
        source_specs=_build_source_specs(config, include_target=True, include_interferer=True),
        fs_hz=float(array_definition.fs_hz),
        sound_speed_m_s=float(array_definition.sound_speed_m_s),
    )
    _, _, target_only_beam_output, _ = _run_fixed_fractional_case(
        config=config,
        case_name="fixed_target_only",
        active_positions_m=active_positions_m,
        source_specs=_build_source_specs(config, include_target=True, include_interferer=False),
        fs_hz=float(array_definition.fs_hz),
        sound_speed_m_s=float(array_definition.sound_speed_m_s),
    )
    _, _, interferer_only_beam_output, _ = _run_fixed_fractional_case(
        config=config,
        case_name="fixed_interferer_only",
        active_positions_m=active_positions_m,
        source_specs=_build_source_specs(config, include_target=False, include_interferer=True),
        fs_hz=float(array_definition.fs_hz),
        sound_speed_m_s=float(array_definition.sound_speed_m_s),
    )

    target_beam_index = int(np.argmin(np.abs(axis_az_deg - float(config.target_azimuth_deg))))
    slc_analysis_block_size = int(config.slc_analysis_block_size)
    slc = BeamDomainSLC(
        n_beam=int(mixed_beam_output.shape[0]),
        fs_hz=float(array_definition.fs_hz),
        block_size=int(slc_analysis_block_size),
        config=slc_config,
    )
    response_matrix = calculate_fractional_beam_response_matrix(
        beamformer=mixed_beamformer,
        frequency_hz=float(config.processing_frequency_hz),
    )
    interferer_frequency_hz = float(config.processing_frequency_hz if config.interferer_frequency_hz is None else config.interferer_frequency_hz)
    interferer_response_matrix = calculate_fractional_beam_response_matrix(
        beamformer=mixed_beamformer,
        frequency_hz=interferer_frequency_hz,
    )
    desired_response_matrix = _build_real_signal_desired_response_matrix(
        response_matrix=response_matrix,
        target_beam_index=target_beam_index,
    )
    # runtime_budget では固定整相や診断図保存ではなく、リアルタイム経路に入る SLC 適用部を測る。
    # mixed beam を複数 block に分け、BeamDomainSLC 内部の共分散を忘却係数付きで積分する。
    slc_output, cancel_output, block_results, slc_result, slc_process_elapsed_sec = _process_streaming_slc_blocks(
        beam_output=mixed_beam_output,
        target_beam_index=target_beam_index,
        slc=slc,
        desired_response_matrix=desired_response_matrix,
        slc_analysis_block_size=slc_analysis_block_size,
    )
    slc_input_duration_sec = float(mixed_beam_output.shape[1]) / float(array_definition.fs_hz)
    enabled_block_count = sum(1 for _, _, result in block_results if result.W is not None)
    disabled_block_count = len(block_results) - int(enabled_block_count)
    fallback_block_count = sum(
        1
        for _, _, result in block_results
        if result.safety is not None and bool(result.safety.fallback_required)
    )
    enabled_results = [result for _, _, result in block_results if result.W is not None]
    # 積分時間が短すぎる場合、block ごとの R_uu がばらつき、条件数や重みノルムも揺れやすい。
    # ここでは後段の memory sweep が安定性を読めるように、block 列の統計量を summary に残す。
    block_condition_numbers = [
        float(result.covariance_condition_number)
        for result in enabled_results
        if result.covariance_condition_number is not None
    ]
    block_weight_norms = [
        float(np.linalg.norm(np.asarray(result.W, dtype=np.complex128)))
        for result in enabled_results
        if result.W is not None
    ]
    block_alphas = [float(result.alpha) for result in enabled_results if result.alpha is not None]
    slc_block_time_sec = float(slc_analysis_block_size) / float(array_definition.fs_hz)
    covariance_memory_summary = _covariance_memory_diagnostics(
        alpha=None if slc_result.alpha is None else float(slc_result.alpha),
        block_size=int(slc_analysis_block_size),
        block_time_sec=slc_block_time_sec,
        memory_time_sec=float(slc_config.memory_time_sec),
        enabled_block_count=int(enabled_block_count),
    )
    fixed_mixed_target = mixed_beam_output[target_beam_index, :]
    fixed_target_component = target_only_beam_output[target_beam_index, :]
    fixed_interferer_component = interferer_only_beam_output[target_beam_index, :]
    raw_candidate_eta = float(slc_config.eta_normal)

    # SLC 係数は複素最小二乗で求まるため、出力も複素になり得る。
    # raw candidate は safety gate 前の方式評価、effective は safety gate 後の運用出力として分けて記録する。
    raw_mixed_target = np.asarray(fixed_mixed_target - raw_candidate_eta * np.asarray(cancel_output[0]), dtype=np.complex128)
    effective_mixed_target = np.asarray(slc_output[0], dtype=np.complex128)
    raw_target_component = _apply_streaming_slc_to_component(
        target_only_beam_output,
        target_beam_index,
        block_results,
        eta_override=raw_candidate_eta,
    )
    raw_interferer_component = _apply_streaming_slc_to_component(
        interferer_only_beam_output,
        target_beam_index,
        block_results,
        eta_override=raw_candidate_eta,
    )
    effective_target_component = _apply_streaming_slc_to_component(target_only_beam_output, target_beam_index, block_results)
    effective_interferer_component = _apply_streaming_slc_to_component(interferer_only_beam_output, target_beam_index, block_results)
    level_summary = {
        "mixed_before_db20": _calculate_input_rms_level(fixed_mixed_target),
        "mixed_after_raw_slc_db20": _calculate_input_rms_level(raw_mixed_target),
        "mixed_after_effective_db20": _calculate_input_rms_level(effective_mixed_target),
        "target_before_db20": _calculate_input_rms_level(fixed_target_component),
        "target_after_raw_slc_db20": _calculate_input_rms_level(raw_target_component),
        "target_after_effective_db20": _calculate_input_rms_level(effective_target_component),
        "interferer_before_db20": _calculate_input_rms_level(fixed_interferer_component),
        "interferer_after_raw_slc_db20": _calculate_input_rms_level(raw_interferer_component),
        "interferer_after_effective_db20": _calculate_input_rms_level(effective_interferer_component),
    }
    level_summary["raw_mixed_power_delta_db"] = float(level_summary["mixed_after_raw_slc_db20"] - level_summary["mixed_before_db20"])
    level_summary["effective_mixed_power_delta_db"] = float(level_summary["mixed_after_effective_db20"] - level_summary["mixed_before_db20"])
    level_summary["raw_target_power_delta_db"] = float(level_summary["target_after_raw_slc_db20"] - level_summary["target_before_db20"])
    level_summary["effective_target_power_delta_db"] = float(level_summary["target_after_effective_db20"] - level_summary["target_before_db20"])
    level_summary["raw_interferer_reduction_db"] = float(level_summary["interferer_before_db20"] - level_summary["interferer_after_raw_slc_db20"])
    level_summary["effective_interferer_reduction_db"] = float(level_summary["interferer_before_db20"] - level_summary["interferer_after_effective_db20"])

    safety_fallback_required = bool(fallback_block_count > 0)

    plot_path = output_dir / "target_leakage_levels.png"
    _plot_target_leakage_levels(
        output_path=plot_path,
        levels={
            "mixed before": float(level_summary["mixed_before_db20"]),
            "mixed raw SLC": float(level_summary["mixed_after_raw_slc_db20"]),
            "mixed effective": float(level_summary["mixed_after_effective_db20"]),
            "interf before": float(level_summary["interferer_before_db20"]),
            "interf raw SLC": float(level_summary["interferer_after_raw_slc_db20"]),
            "interf effective": float(level_summary["interferer_after_effective_db20"]),
        },
        title="Target-beam leakage before/after time-domain SLC",
    )

    waveform_overlay_path = output_dir / "slc_before_after_waveform_overlay.png"
    _plot_slc_waveform_overlay(
        output_path=waveform_overlay_path,
        fixed_mixed_target=fixed_mixed_target,
        raw_mixed_target=raw_mixed_target,
        effective_mixed_target=effective_mixed_target,
        fs_hz=float(array_definition.fs_hz),
    )
    spectrum_overlay_path = output_dir / "slc_component_spectrum_overlay.png"
    _plot_slc_component_spectrum_overlay(
        output_path=spectrum_overlay_path,
        fixed_target_component=fixed_target_component,
        raw_target_component=raw_target_component,
        fixed_interferer_component=fixed_interferer_component,
        raw_interferer_component=raw_interferer_component,
        fs_hz=float(array_definition.fs_hz),
        target_frequency_hz=float(config.processing_frequency_hz),
        interferer_frequency_hz=interferer_frequency_hz,
    )

    protected_bl_levels = _calculate_protected_target_slc_bl_levels(
        target_response_matrix=response_matrix,
        interferer_response_matrix=interferer_response_matrix,
        target_beam_index=target_beam_index,
        slc_result=slc_result,
        slc_config=slc_config,
        fs_hz=float(array_definition.fs_hz),
        target_frequency_hz=float(config.processing_frequency_hz),
        interferer_frequency_hz=interferer_frequency_hz,
        target_level_db20=float(config.target_level_db20),
        interferer_level_db20=float(config.interferer_level_db20),
    )
    target_response_bl_overlay_path = output_dir / "protected_target_response_bl_overlay.png"
    _plot_slc_bl_overlay(
        output_path=target_response_bl_overlay_path,
        axis_az_deg=axis_az_deg,
        before_levels_db20=protected_bl_levels.target_before_db20,
        after_levels_db20=protected_bl_levels.target_after_db20,
        marker_azimuth_deg=float(config.target_azimuth_deg),
        marker_label="Target source azimuth",
        title="Protected target-beam BL overlay at target frequency",
        caption=(
            "target-centric time-domain SLC, L=1。最終有効 SLC 係数を固定した source-response。"
            "streaming 成分別 RMS とは別評価。target 周波数で mainlobe 保護を見る。"
        ),
        before_label="fixed target beam",
        after_label="SLC target beam",
    )
    interferer_response_bl_overlay_path = output_dir / "protected_target_interferer_response_bl_overlay.png"
    _plot_slc_bl_overlay(
        output_path=interferer_response_bl_overlay_path,
        axis_az_deg=axis_az_deg,
        before_levels_db20=protected_bl_levels.interferer_before_db20,
        after_levels_db20=protected_bl_levels.interferer_after_db20,
        marker_azimuth_deg=float(config.interferer_azimuth_deg),
        marker_label="Interferer source azimuth",
        title="Protected target-beam BL overlay at interferer frequency",
        caption=(
            "target-centric time-domain SLC, L=1。最終有効 SLC 係数を固定した source-response。"
            "streaming 成分別 RMS とは別評価。interferer 周波数で marker / guard 外応答を見る。"
        ),
        before_label="fixed target beam",
        after_label="SLC target beam",
    )
    target_source_index = int(np.argmin(np.abs(axis_az_deg - float(config.target_azimuth_deg))))
    interferer_source_index = int(np.argmin(np.abs(axis_az_deg - float(config.interferer_azimuth_deg))))
    # SLC の方式判断では「target が悪化しない」だけでは不足する。
    # guard 外 peak の改善量と最大悪化量、既知干渉方位 marker の低下量を分けて、
    # BL 図上の sidelobe 改善が実際に得られているかを数値で判定する。
    target_frequency_sidelobe_metrics = _protected_target_bl_sidelobe_metrics(
        axis_az_deg=axis_az_deg,
        before_levels_db20=protected_bl_levels.target_before_db20,
        after_levels_db20=protected_bl_levels.target_after_db20,
        target_beam_index=int(target_beam_index),
        marker_azimuth_deg=float(config.target_azimuth_deg),
        guard_beam_count=int(slc_config.guard),
    )
    interferer_frequency_sidelobe_metrics = _protected_target_bl_sidelobe_metrics(
        axis_az_deg=axis_az_deg,
        before_levels_db20=protected_bl_levels.interferer_before_db20,
        after_levels_db20=protected_bl_levels.interferer_after_db20,
        target_beam_index=int(target_beam_index),
        marker_azimuth_deg=float(config.interferer_azimuth_deg),
        guard_beam_count=int(slc_config.guard),
    )
    # BL 図の方式判断では、干渉 marker の一点が落ちても、guard 外 peak が下がらなければ改善とは判定しない。
    # `max_guard_outside_worsening_db <= 0` も要求し、別方位へ sidelobe を押し出しただけの方式を不合格にする。
    interferer_guard_peak_delta_db = float(interferer_frequency_sidelobe_metrics["guard_outside_peak_delta_db"])
    interferer_max_worsening_db = float(interferer_frequency_sidelobe_metrics["max_guard_outside_worsening_db"])
    interferer_first_sidelobe_reduction_db = float(interferer_frequency_sidelobe_metrics["first_sidelobe_reduction_db"])
    bl_improvement_pass = bool(
        interferer_guard_peak_delta_db < 0.0
        and interferer_max_worsening_db <= 0.0
        and interferer_first_sidelobe_reduction_db > 0.0
    )
    bl_improvement_failure_reason = (
        "none"
        if bl_improvement_pass
        else "guard_outside_peak_or_first_sidelobe_not_reduced_or_local_worsening_detected"
    )
    protected_bl_summary = {
        "definition": "protected target beam fixed; x-axis is source azimuth, not output beam index",
        "bl_improvement_pass": bl_improvement_pass,
        "bl_improvement_failure_reason": bl_improvement_failure_reason,
        "target_source_azimuth_deg": float(axis_az_deg[target_source_index]),
        "interferer_source_azimuth_deg": float(axis_az_deg[interferer_source_index]),
        "target_frequency_before_at_target_db20": float(protected_bl_levels.target_before_db20[target_source_index]),
        "target_frequency_after_at_target_db20": float(protected_bl_levels.target_after_db20[target_source_index]),
        "target_frequency_delta_at_target_db": float(
            protected_bl_levels.target_after_db20[target_source_index] - protected_bl_levels.target_before_db20[target_source_index]
        ),
        "interferer_frequency_before_at_interferer_db20": float(protected_bl_levels.interferer_before_db20[interferer_source_index]),
        "interferer_frequency_after_at_interferer_db20": float(protected_bl_levels.interferer_after_db20[interferer_source_index]),
        "interferer_frequency_reduction_at_interferer_db": float(
            protected_bl_levels.interferer_before_db20[interferer_source_index]
            - protected_bl_levels.interferer_after_db20[interferer_source_index]
        ),
        "interferer_frequency_before_at_target_db20": float(protected_bl_levels.interferer_before_db20[target_source_index]),
        "interferer_frequency_after_at_target_db20": float(protected_bl_levels.interferer_after_db20[target_source_index]),
        "interferer_frequency_reduction_at_target_db": float(
            protected_bl_levels.interferer_before_db20[target_source_index]
            - protected_bl_levels.interferer_after_db20[target_source_index]
        ),
        "target_frequency_sidelobe_metrics": target_frequency_sidelobe_metrics,
        "interferer_frequency_sidelobe_metrics": interferer_frequency_sidelobe_metrics,
    }
    summary: dict[str, object] = {
        "array_definition_path": str(Path(config.operational_array_definition_path).resolve()),
        "fractional_delay_filter_bank_path": str(Path(config.fractional_delay_filter_bank_path).resolve()),
        "fixed_mixed_summary_path": _resolve_summary_path_or_default(
            fixed_mixed_summary,
            key="fixed_summary_path",
            default_path=output_dir / "fixed_mixed" / "summary.json",
        ),
        "processing_frequency_hz": float(config.processing_frequency_hz),
        "target_frequency_hz": float(config.processing_frequency_hz),
        "interferer_frequency_hz": interferer_frequency_hz,
        "target_level_db20": float(config.target_level_db20),
        "interferer_level_db20": float(config.interferer_level_db20),
        "target_azimuth_deg": float(config.target_azimuth_deg),
        "interferer_azimuth_deg": float(config.interferer_azimuth_deg),
        "target_beam_index": int(target_beam_index),
        "target_beam_azimuth_deg": float(axis_az_deg[target_beam_index]),
        "n_beam": int(mixed_beam_output.shape[0]),
        "n_sample": int(mixed_beam_output.shape[1]),
        "active_channel_count": int(active_indices.size),
        "active_aperture_m": active_aperture_m,
        "slc_config": {
            "guard": int(slc_config.guard),
            "loading": float(slc_config.loading),
            "loading_reference": "mean diagonal reference covariance power",
            "memory_time_sec": float(slc_config.memory_time_sec),
            "heading_scale_deg": float(slc_config.heading_scale_deg),
            "min_ref": int(slc_config.min_ref),
            "sample_per_dof": float(slc_config.sample_per_dof),
            "tap_len": int(slc_config.tap_len),
            "eta_normal": float(slc_config.eta_normal),
            "eta_limited": float(slc_config.eta_limited),
            "enable_heading_forgetting": bool(slc_config.enable_heading_forgetting),
            "enable_output_safety_gate": bool(slc_config.enable_output_safety_gate),
            "max_output_power_increase_db": float(slc_config.max_output_power_increase_db),
            "max_output_power_drop_db": float(slc_config.max_output_power_drop_db),
            "max_cancel_power_relative_db": float(slc_config.max_cancel_power_relative_db),
        },
        "slc_process": {
            "mode": str(slc_result.mode),
            "eta": float(slc_result.eta),
            "raw_candidate_eta": raw_candidate_eta,
            "alpha": None if slc_result.alpha is None else float(slc_result.alpha),
            "reference_beam_count": int(slc_result.reference_beams.size),
            "capacity": slc_result.capacity.as_dict(),
            "weight_norm": float(np.linalg.norm(np.asarray(slc_result.W, dtype=np.complex128))),
            "condition_number": None
            if slc_result.covariance_condition_number is None
            else float(slc_result.covariance_condition_number),
            "condition_number_matrix": "R_uu + loading * mean(diag(R_uu)) I",
            "covariance_integration": "exponential_forgetting_by_block",
            "analysis_block_size": int(slc_analysis_block_size),
            "covariance_memory": covariance_memory_summary,
            "block_condition_number_stats": _finite_float_statistics(block_condition_numbers),
            "block_weight_norm_stats": _finite_float_statistics(block_weight_norms),
            "block_alpha_stats": _finite_float_statistics(block_alphas),
            "block_count": int(len(block_results)),
            "enabled_block_count": int(enabled_block_count),
            "disabled_block_count": int(disabled_block_count),
            "fallback_block_count": int(fallback_block_count),
            "elapsed_sec": float(slc_process_elapsed_sec),
            "input_duration_sec": float(slc_input_duration_sec),
            "realtime_factor": float(slc_process_elapsed_sec / slc_input_duration_sec),
            "uses_desired_response_blocking": bool(slc_result.reference_blocking_matrix is not None),
            "safety": None if slc_result.safety is None else slc_result.safety.as_dict(),
        },
        "level_reference": "dB re input RMS",
        "levels": level_summary,
        "raw_slc_reduces_interferer": bool(level_summary["raw_interferer_reduction_db"] > 0.0),
        "slc_bl_improvement_pass": bl_improvement_pass,
        "safety_fallback_required": safety_fallback_required,
        "recommended_output": "raw_slc" if (not safety_fallback_required and bl_improvement_pass) else "fixed_beamformer",
        "target_leakage_levels_png_path": str(plot_path.resolve()),
        "slc_before_after_waveform_overlay_png_path": str(waveform_overlay_path.resolve()),
        "slc_component_spectrum_overlay_png_path": str(spectrum_overlay_path.resolve()),
        "protected_target_response_bl_overlay_png_path": str(target_response_bl_overlay_path.resolve()),
        "protected_target_interferer_response_bl_overlay_png_path": str(interferer_response_bl_overlay_path.resolve()),
        "protected_target_bl_summary": protected_bl_summary,
    }
    (output_dir / "time_domain_slc_leakage_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


__all__ = [
    "OperationalTimeDomainSlcDiagnosticConfig",
    "run_operational_time_domain_slc_leakage_diagnostics",
]
