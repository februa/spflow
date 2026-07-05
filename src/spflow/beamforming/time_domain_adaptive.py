"""時間領域 MVDR / LCMV / GSC の共通部品を実装するモジュール。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .._validation import require, require_non_negative_float, require_positive_float, require_positive_int


@dataclass(frozen=True)
class TimeDomainAdaptiveWeightDiagnostics:
    """時間領域適応重みの数値健全性を保持する。

    このクラスは、channel×tap 空間で設計した MVDR / LCMV / GSC 重みについて、
    対角ロード後の共分散条件数、自由度、制約数を記録する。

    入力は設計済みの共分散行列と制約行列から得た scalar 診断値であり、出力は JSON 化しやすい辞書である。
    重み設計、波形への適用、BL/FRAZ/BTR 描画は責務に含めない。
    信号処理上は、時間領域適応方式を SLC と同じ評価基準で比較するための covariance health 診断に位置づく。
    """

    degree_of_freedom: int
    constraint_count: int
    output_count: int
    diagonal_loading: float
    loaded_condition_number: float

    def as_dict(self) -> dict[str, int | float]:
        """診断値を JSON へ保存しやすい辞書へ変換する。

        Returns:
            自由度、制約数、出力数、対角ロード量、条件数を含む辞書。
            条件数は無次元比、対角ロード量は共分散平均対角 power に対する比である。
        """
        return {
            "degree_of_freedom": int(self.degree_of_freedom),
            "constraint_count": int(self.constraint_count),
            "output_count": int(self.output_count),
            "diagonal_loading": float(self.diagonal_loading),
            "loaded_condition_number": float(self.loaded_condition_number),
        }


def build_time_tapped_snapshot_matrix(channel_signals: NDArray[Any], tap_len: int) -> NDArray[np.complex128]:
    """channel 信号を channel×tap の snapshot 行列へ展開する。

    Args:
        channel_signals: 入力信号。shape は `[n_ch, n_sample]`。
            axis=0 が sensor channel、axis=1 が時間 sample である。
        tap_len: FIR tap 数 `L`。単位は sample。

    Returns:
        時間タップ付き snapshot。shape は `[n_ch * L, n_sample - L + 1]`。
        row は `lag=0, 1, ..., L-1` の順に channel 軸を積む。

    Raises:
        ValueError: 入力が 2 次元でない、tap_len が正でない、または full tap を作れない場合。

    境界条件:
        先頭 `L-1` sample は過去 sample が不足するため、この行列には含めない。
        出力波形へ戻す関数では、その区間を 0 で埋めて「重み未適用」と明示する。
    """
    signals = np.asarray(channel_signals, dtype=np.complex128)
    require(signals.ndim == 2, "channel_signals must have shape (n_ch, n_sample).")
    require_positive_int("tap_len", int(tap_len))
    require(signals.shape[1] >= int(tap_len), "channel_signals must contain at least tap_len samples.")

    n_ch = int(signals.shape[0])
    n_sample = int(signals.shape[1])
    n_valid_sample = n_sample - int(tap_len) + 1
    tapped = np.zeros((n_ch * int(tap_len), n_valid_sample), dtype=np.complex128)
    for lag_index in range(int(tap_len)):
        row_start = lag_index * n_ch
        row_stop = row_start + n_ch
        sample_start = int(tap_len) - 1 - lag_index
        sample_stop = sample_start + n_valid_sample

        # X_tap[lag, ch, n] = x[ch, n + L - 1 - lag]。
        # lag=0 は現在 sample、lag>0 は過去 sample であり、FIR 重み w[lag, ch] と対応する。
        tapped[row_start:row_stop, :] = signals[:, sample_start:sample_stop]
    return tapped


def estimate_time_domain_covariance(tapped_snapshots: NDArray[Any]) -> NDArray[np.complex128]:
    """channel×tap snapshot から時間領域共分散を推定する。

    Args:
        tapped_snapshots: 時間タップ付き snapshot。shape は `[n_dof, n_snapshot]`。
            axis=0 が channel×tap 自由度、axis=1 が時間 snapshot である。

    Returns:
        共分散行列 `R = X X^H / K`。shape は `[n_dof, n_dof]`。
        値の単位は入力信号 power に対応する。

    Raises:
        ValueError: 入力が 2 次元でない、または snapshot が空の場合。
    """
    snapshots = np.asarray(tapped_snapshots, dtype=np.complex128)
    require(snapshots.ndim == 2, "tapped_snapshots must have shape (n_dof, n_snapshot).")
    require(snapshots.shape[1] > 0, "tapped_snapshots must contain at least one snapshot.")

    # R[dof_i, dof_j] = (1/K) Σ_k X[dof_i, k] conj(X[dof_j, k])。
    # MVDR / LCMV はこの R に対して出力 power w^H R w を最小化する。
    return np.asarray((snapshots @ snapshots.conj().T) / float(snapshots.shape[1]), dtype=np.complex128)


def build_time_domain_tone_constraint_vector(
    steering_vector: NDArray[Any],
    *,
    frequency_hz: float,
    fs_hz: float,
    tap_len: int,
) -> NDArray[np.complex128]:
    """複素 tone の channel×tap 制約ベクトルを作る。

    Args:
        steering_vector: 該当方位・周波数の channel steering。shape は `[n_ch]`。
            axis=0 が sensor channel である。
        frequency_hz: tone 周波数。単位は Hz。
        fs_hz: サンプリング周波数。単位は Hz。
        tap_len: FIR tap 数 `L`。単位は sample。

    Returns:
        制約ベクトル。shape は `[n_ch * L]`。
        `w^H c = desired_response` が、その tone に対する歪みなし条件である。

    Raises:
        ValueError: steering が 1 次元でない、周波数や fs が不正、tap_len が正でない場合。

    Notes:
        入力 tone を `x_ch[n] = a_ch exp(j 2π f n / fs)` とすると、
        lag 付き snapshot は `a_ch exp(-j 2π f lag / fs) exp(j 2π f n / fs)` になる。
        そのため制約ベクトルには `exp(-j 2π f lag / fs)` の位相を掛ける。
    """
    steering = np.asarray(steering_vector, dtype=np.complex128)
    require(steering.ndim == 1, "steering_vector must have shape (n_ch,).")
    require(steering.size > 0, "steering_vector must contain at least one channel.")
    require_positive_float("frequency_hz", float(frequency_hz))
    require_positive_float("fs_hz", float(fs_hz))
    require_positive_int("tap_len", int(tap_len))

    constraints: list[NDArray[np.complex128]] = []
    for lag_index in range(int(tap_len)):
        # 過去 sample `n-lag` の tone 位相は、現在 sample `n` に対してこの分だけ遅れる。
        lag_phase = np.exp(-1j * 2.0 * np.pi * float(frequency_hz) * float(lag_index) / float(fs_hz))
        constraints.append(np.asarray(steering * lag_phase, dtype=np.complex128))
    return np.concatenate(constraints, axis=0).astype(np.complex128)


def build_real_tone_constraint_matrix(
    steering_vector: NDArray[Any],
    *,
    frequency_hz: float,
    fs_hz: float,
    tap_len: int,
) -> NDArray[np.complex128]:
    """実 tone を保護するための正負周波数制約行列を作る。

    Args:
        steering_vector: 正周波数側の channel steering。shape は `[n_ch]`。
        frequency_hz: tone 周波数。単位は Hz。
        fs_hz: サンプリング周波数。単位は Hz。
        tap_len: FIR tap 数 `L`。単位は sample。

    Returns:
        正周波数と負周波数の制約行列。shape は `[n_ch * L, 2]`。
        axis=0 が channel×tap 自由度、axis=1 が `[+f, -f]` 制約である。

    Notes:
        実信号は正負周波数の共役対を持つ。正周波数だけを制約すると、
        負周波数側で target が変形する余地が残るため、実信号 target 保護では `c` と `conj(c)` を同時に制約する。
    """
    positive_constraint = build_time_domain_tone_constraint_vector(
        steering_vector,
        frequency_hz=float(frequency_hz),
        fs_hz=float(fs_hz),
        tap_len=int(tap_len),
    )
    return np.stack([positive_constraint, positive_constraint.conj()], axis=1).astype(np.complex128)


def _loaded_covariance(covariance: NDArray[Any], diagonal_loading: float) -> NDArray[np.complex128]:
    """平均対角 power に対する比で対角ロードを加えた共分散を返す。"""
    covariance_matrix = np.asarray(covariance, dtype=np.complex128)
    require(covariance_matrix.ndim == 2, "covariance must have shape (n_dof, n_dof).")
    require(covariance_matrix.shape[0] == covariance_matrix.shape[1], "covariance must be square.")
    require_non_negative_float("diagonal_loading", float(diagonal_loading))

    n_dof = int(covariance_matrix.shape[0])
    if float(diagonal_loading) == 0.0:
        return covariance_matrix.copy()

    # 対角ロードは絶対 power ではなく平均対角 power に対する比とする。
    # 入力レベルが変わっても正則化の相対強度を保ち、少数 snapshot や高相関 source での重み発散を抑える。
    average_power = float(np.real(np.trace(covariance_matrix)) / float(n_dof))
    loading_power = float(diagonal_loading) * (average_power if average_power > 0.0 else 1.0)
    return covariance_matrix + loading_power * np.eye(n_dof, dtype=np.complex128)


def design_time_domain_lcmv_weights(
    covariance: NDArray[Any],
    constraint_matrix: NDArray[Any],
    desired_response: NDArray[Any],
    *,
    diagonal_loading: float = 1.0e-3,
) -> NDArray[np.complex128]:
    """時間領域 LCMV 重みを設計する。

    Args:
        covariance: channel×tap 共分散。shape は `[n_dof, n_dof]`。
            `n_dof = n_ch * tap_len` である。
        constraint_matrix: 制約行列 `C`。shape は `[n_dof, n_constraint]`。
            axis=0 が channel×tap 自由度、axis=1 が保護または null 制約である。
        desired_response: 制約応答 `f`。shape は `[n_constraint]` または `[n_constraint, n_output]`。
            例として target 保護は 1、干渉 null は 0 を指定する。
        diagonal_loading: 平均対角 power に対する対角ロード比。無次元。

    Returns:
        LCMV 重み。shape は `[n_dof, n_output]`。
        適用時は `y = w^H x_tap` として使う。

    Raises:
        ValueError: shape が整合しない場合。
        numpy.linalg.LinAlgError: 対角ロード後の制約 Gram 行列が解けない場合。

    Notes:
        LCMV は `min w^H R w subject to C^H w = f` を解く。
        解は `w = R^{-1} C (C^H R^{-1} C)^{-1} f` である。
    """
    loaded = _loaded_covariance(covariance, diagonal_loading=float(diagonal_loading))
    constraints = np.asarray(constraint_matrix, dtype=np.complex128)
    response = np.asarray(desired_response, dtype=np.complex128)
    require(constraints.ndim == 2, "constraint_matrix must have shape (n_dof, n_constraint).")
    require(constraints.shape[0] == loaded.shape[0], "constraint_matrix and covariance must agree on n_dof.")
    require(constraints.shape[1] > 0, "constraint_matrix must contain at least one constraint.")
    if response.ndim == 1:
        response = response[:, np.newaxis]
    require(response.ndim == 2, "desired_response must have shape (n_constraint,) or (n_constraint, n_output).")
    require(response.shape[0] == constraints.shape[1], "desired_response and constraint_matrix must agree on n_constraint.")

    # solve(R, C) により R^{-1}C を直接求め、明示的な逆行列を作らない。
    # これは MVDR / LCMV の標準形 `R^{-1}C(C^H R^{-1}C)^{-1}f` に対応する。
    inverse_covariance_constraints = np.linalg.solve(loaded, constraints)
    constraint_gram = constraints.conj().T @ inverse_covariance_constraints
    lagrange_solution = np.linalg.solve(constraint_gram, response)
    return np.asarray(inverse_covariance_constraints @ lagrange_solution, dtype=np.complex128)


def design_time_domain_mvdr_weights(
    covariance: NDArray[Any],
    target_constraint_vector: NDArray[Any],
    *,
    diagonal_loading: float = 1.0e-3,
) -> NDArray[np.complex128]:
    """単一 target 制約の時間領域 MVDR 重みを設計する。

    Args:
        covariance: channel×tap 共分散。shape は `[n_dof, n_dof]`。
        target_constraint_vector: target tone または target 応答の制約ベクトル。shape は `[n_dof]`。
        diagonal_loading: 平均対角 power に対する対角ロード比。無次元。

    Returns:
        MVDR 重み。shape は `[n_dof, 1]`。
        `w^H target_constraint_vector = 1` を満たす。

    Notes:
        MVDR は LCMV の 1 制約の場合であり、ここでは実装を LCMV に一本化する。
    """
    target_constraint = np.asarray(target_constraint_vector, dtype=np.complex128)
    require(target_constraint.ndim == 1, "target_constraint_vector must have shape (n_dof,).")
    return design_time_domain_lcmv_weights(
        covariance,
        target_constraint[:, np.newaxis],
        np.array([1.0 + 0.0j], dtype=np.complex128),
        diagonal_loading=float(diagonal_loading),
    )


def build_gsc_blocking_matrix(constraint_matrix: NDArray[Any], *, rcond: float = 1.0e-10) -> NDArray[np.complex128]:
    """LCMV 制約を満たす nullspace blocking matrix を作る。

    Args:
        constraint_matrix: LCMV 制約行列 `C`。shape は `[n_dof, n_constraint]`。
            GSC の blocking matrix は `C^H B = 0` を満たす。
        rcond: SVD rank 判定の相対閾値。無次元。

    Returns:
        blocking matrix `B`。shape は `[n_dof, n_block]`。
        `n_block = n_dof - rank(C^H)` である。

    Raises:
        ValueError: 入力 shape または rcond が不正な場合。

    Notes:
        SVD を使うのは、複数制約が近接方位や正負周波数で線形従属に近くなる場合があるためである。
        rank 落ち時も数値 rank に基づく nullspace を使い、GSC の自由度を過大評価しない。
    """
    constraints = np.asarray(constraint_matrix, dtype=np.complex128)
    require(constraints.ndim == 2, "constraint_matrix must have shape (n_dof, n_constraint).")
    require(constraints.shape[0] > 0, "constraint_matrix must contain at least one dof.")
    require(constraints.shape[1] > 0, "constraint_matrix must contain at least one constraint.")
    require_positive_float("rcond", float(rcond))

    _, singular_values, vh = np.linalg.svd(constraints.conj().T, full_matrices=True)
    if singular_values.size == 0:
        rank = 0
    else:
        rank_threshold = float(rcond) * float(np.max(singular_values))
        # np.sum は np.integer を返すため、slice index として使う前に int へ確定する。
        rank = int(np.sum(singular_values > rank_threshold))

    # vh[rank:, :] は `C^H` の右 nullspace の基底を行として持つ。
    # 転置共役して column basis に直すことで、C^H B = 0 を満たす blocking matrix になる。
    return np.asarray(vh[rank:, :].conj().T, dtype=np.complex128)


def design_time_domain_gsc_weights(
    covariance: NDArray[Any],
    constraint_matrix: NDArray[Any],
    desired_response: NDArray[Any],
    *,
    diagonal_loading: float = 1.0e-3,
    rcond: float = 1.0e-10,
) -> NDArray[np.complex128]:
    """GSC 分解で時間領域制約付き最小分散重みを設計する。

    Args:
        covariance: channel×tap 共分散。shape は `[n_dof, n_dof]`。
        constraint_matrix: 制約行列 `C`。shape は `[n_dof, n_constraint]`。
        desired_response: 制約応答 `f`。shape は `[n_constraint]` または `[n_constraint, n_output]`。
        diagonal_loading: 平均対角 power に対する対角ロード比。無次元。
        rcond: blocking matrix を作る SVD rank 判定閾値。無次元。

    Returns:
        GSC 重み。shape は `[n_dof, n_output]`。
        LCMV と同じ `C^H w = f` を満たす。

    Notes:
        GSC は `w = w_q - B g` と分解する。`w_q` は制約を満たす quiescent weight、
        `B` は制約空間を消す blocking matrix、`g` は blocked reference 上の最小分散キャンセラである。
        ここでは LCMV との等価性を確認できるよう、同じ loaded covariance で `g` を解く。
    """
    loaded = _loaded_covariance(covariance, diagonal_loading=float(diagonal_loading))
    constraints = np.asarray(constraint_matrix, dtype=np.complex128)
    response = np.asarray(desired_response, dtype=np.complex128)
    require(constraints.ndim == 2, "constraint_matrix must have shape (n_dof, n_constraint).")
    require(constraints.shape[0] == loaded.shape[0], "constraint_matrix and covariance must agree on n_dof.")
    if response.ndim == 1:
        response = response[:, np.newaxis]
    require(response.ndim == 2, "desired_response must have shape (n_constraint,) or (n_constraint, n_output).")
    require(response.shape[0] == constraints.shape[1], "desired_response and constraint_matrix must agree on n_constraint.")

    # w_q = C(C^H C)^+ f は、制約を満たす最小ノルムの固定重みである。
    # GSC ではこの主経路を保ち、blocking 後の自由度だけで出力 power を下げる。
    constraint_gram = constraints.conj().T @ constraints
    quiescent_weights = constraints @ np.linalg.pinv(constraint_gram) @ response
    blocking_matrix = build_gsc_blocking_matrix(constraints, rcond=float(rcond))
    if blocking_matrix.shape[1] == 0:
        return np.asarray(quiescent_weights, dtype=np.complex128)

    blocked_covariance = blocking_matrix.conj().T @ loaded @ blocking_matrix
    blocked_cross = blocking_matrix.conj().T @ loaded @ quiescent_weights
    adaptive_weights = np.linalg.solve(blocked_covariance, blocked_cross)
    return np.asarray(quiescent_weights - blocking_matrix @ adaptive_weights, dtype=np.complex128)


def apply_time_domain_fir_beamformer(
    channel_signals: NDArray[Any],
    weights: NDArray[Any],
    *,
    tap_len: int,
) -> NDArray[np.complex128]:
    """channel×tap FIR ビームフォーマ重みを時間波形へ適用する。

    Args:
        channel_signals: 入力信号。shape は `[n_ch, n_sample]`。
        weights: FIR 重み。shape は `[n_ch * tap_len]` または `[n_ch * tap_len, n_output]`。
        tap_len: FIR tap 数 `L`。単位は sample。

    Returns:
        出力信号。shape は `[n_output, n_sample]`。
        先頭 `L-1` sample は full tap が揃わないため 0 とする。

    Raises:
        ValueError: 入力 shape が整合しない場合。
    """
    tapped_snapshots = build_time_tapped_snapshot_matrix(channel_signals, tap_len=int(tap_len))
    beam_weights = np.asarray(weights, dtype=np.complex128)
    if beam_weights.ndim == 1:
        beam_weights = beam_weights[:, np.newaxis]
    require(beam_weights.ndim == 2, "weights must have shape (n_dof,) or (n_dof, n_output).")
    require(beam_weights.shape[0] == tapped_snapshots.shape[0], "weights and channel_signals must agree on n_ch * tap_len.")

    n_sample = int(np.asarray(channel_signals).shape[1])
    output = np.zeros((beam_weights.shape[1], n_sample), dtype=np.complex128)
    # y[out, n] = Σ_dof conj(w[dof, out]) X_tap[dof, n]。
    # LCMV / MVDR の制約 `w^H c = f` と同じ内積向きで時間波形へ適用する。
    valid_output = beam_weights.conj().T @ tapped_snapshots
    output[:, int(tap_len) - 1 :] = valid_output
    return output


def evaluate_constraint_response(weights: NDArray[Any], constraint_matrix: NDArray[Any]) -> NDArray[np.complex128]:
    """設計重みが制約へ与える応答 `C^H W` を計算する。

    Args:
        weights: 設計済み重み。shape は `[n_dof]` または `[n_dof, n_output]`。
        constraint_matrix: 制約行列。shape は `[n_dof, n_constraint]`。

    Returns:
        制約応答。shape は `[n_constraint, n_output]`。
        target 保護なら 1、null 制約なら 0 に近いことを確認する。
    """
    beam_weights = np.asarray(weights, dtype=np.complex128)
    if beam_weights.ndim == 1:
        beam_weights = beam_weights[:, np.newaxis]
    constraints = np.asarray(constraint_matrix, dtype=np.complex128)
    require(beam_weights.ndim == 2, "weights must have shape (n_dof,) or (n_dof, n_output).")
    require(constraints.ndim == 2, "constraint_matrix must have shape (n_dof, n_constraint).")
    require(beam_weights.shape[0] == constraints.shape[0], "weights and constraint_matrix must agree on n_dof.")
    return np.asarray(constraints.conj().T @ beam_weights, dtype=np.complex128)


def diagnose_time_domain_adaptive_weights(
    covariance: NDArray[Any],
    constraint_matrix: NDArray[Any],
    weights: NDArray[Any],
    *,
    diagonal_loading: float,
) -> TimeDomainAdaptiveWeightDiagnostics:
    """時間領域適応重みの covariance health 診断を作る。

    Args:
        covariance: channel×tap 共分散。shape は `[n_dof, n_dof]`。
        constraint_matrix: 制約行列。shape は `[n_dof, n_constraint]`。
        weights: 設計済み重み。shape は `[n_dof]` または `[n_dof, n_output]`。
        diagonal_loading: 平均対角 power に対する対角ロード比。無次元。

    Returns:
        自由度、制約数、出力数、対角ロード、loaded covariance 条件数を含む診断。
    """
    loaded = _loaded_covariance(covariance, diagonal_loading=float(diagonal_loading))
    constraints = np.asarray(constraint_matrix, dtype=np.complex128)
    beam_weights = np.asarray(weights, dtype=np.complex128)
    if beam_weights.ndim == 1:
        beam_weights = beam_weights[:, np.newaxis]
    require(constraints.ndim == 2, "constraint_matrix must have shape (n_dof, n_constraint).")
    require(beam_weights.ndim == 2, "weights must have shape (n_dof,) or (n_dof, n_output).")
    require(constraints.shape[0] == loaded.shape[0], "constraint_matrix and covariance must agree on n_dof.")
    require(beam_weights.shape[0] == loaded.shape[0], "weights and covariance must agree on n_dof.")

    return TimeDomainAdaptiveWeightDiagnostics(
        degree_of_freedom=int(loaded.shape[0]),
        constraint_count=int(constraints.shape[1]),
        output_count=int(beam_weights.shape[1]),
        diagonal_loading=float(diagonal_loading),
        loaded_condition_number=float(np.linalg.cond(loaded)),
    )