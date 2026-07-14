"""source maskを使うbeamforming後のnon-source leakage cancellationを実装する。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .._validation import (
    require,
    require_non_negative_float,
    require_positive_float,
    require_positive_int,
)
from ..beamforming_evaluation.abf_like_metrics import SourceSectorMask
from .beam_domain import (
    BlockLeastSquaresSlcSolver,
    SlcReferenceCapacityChecker,
    SlcReferenceCapacityDecision,
    build_time_tapped_reference_matrix,
)


@dataclass(frozen=True)
class SourceMaskSlcConfig:
    """source-mask SLC の処理条件を保持する。

    このクラスは、source reference beam の取り方、時間 tap 数、対角 loading、eta、
    参照容量判定、安全 gate の閾値をまとめて保持する。

    入力は固定整相後 `beam_output[beam, sample]` に対して使う scalar 設定であり、
    出力は `SourceMaskNonSourceLeakageSubtractor` が参照する不変設定である。

    固定整相そのもの、source 検出、BL / FRAZ / BTR 描画、FIR 係数設計は責務に含めない。
    信号処理上は、source beam を保護しながら non-source sector の source-correlated leakage だけを
    差し引く後段 SLC の制御パラメータに位置づく。
    """

    eta: float = 0.5
    loading: float = 3.0e-2
    tap_len: int = 1
    source_reference_guard_beam_count: int = 0
    min_ref: int = 1
    sample_per_dof: float = 5.0
    condition_number_limit: float = 1.0e8
    weight_norm_limit: float | None = None
    copy_source_beams: bool = True

    def __post_init__(self) -> None:
        """設定値の範囲を検証する。"""
        require(0.0 <= float(self.eta) <= 1.0, "eta must lie in [0.0, 1.0].")
        require_non_negative_float("loading", float(self.loading))
        require_positive_int("tap_len", int(self.tap_len))
        require_non_negative_float(
            "source_reference_guard_beam_count",
            float(self.source_reference_guard_beam_count),
        )
        require_positive_int("min_ref", int(self.min_ref))
        require_positive_float("sample_per_dof", float(self.sample_per_dof))
        require_positive_float("condition_number_limit", float(self.condition_number_limit))
        if self.weight_norm_limit is not None:
            require_positive_float("weight_norm_limit", float(self.weight_norm_limit))


@dataclass(frozen=True)
class SourceMaskSlcHealth:
    """source-mask SLC の健全性と fallback 状態を保持する。

    このクラスは、参照容量、loaded covariance 条件数、重みノルム、NaN/inf 個数、
    safety fallback 要否を summary として保持する。

    入力は SLC 実行中に得られた scalar 診断量であり、出力は方式比較 JSON に保存可能な
    診断辞書である。

    SLC 係数の計算や出力波形の生成は責務に含めない。
    信号処理上は、raw candidate と fixed fallback 後の effective output を分けて読むための
    運用安全診断に位置づく。
    """

    mode: str
    capacity: SlcReferenceCapacityDecision
    eta: float
    loading: float
    tap_len: int
    n_ref: int
    n_non_source: int
    condition_number: float | None
    weight_norm: float | None
    nan_inf_count: int
    safety_fallback_required: bool
    failure_reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        """JSON summary に保存しやすい辞書へ変換する。"""
        return {
            "mode": self.mode,
            "capacity": self.capacity.as_dict(),
            "eta": float(self.eta),
            "loading": float(self.loading),
            "loading_reference": "mean diagonal source-reference covariance power",
            "tap_len": int(self.tap_len),
            "n_ref": int(self.n_ref),
            "n_non_source": int(self.n_non_source),
            "condition_number": None
            if self.condition_number is None
            else float(self.condition_number),
            "condition_number_matrix": "R_ss + loading * mean(diag(R_ss)) I",
            "weight_norm": None if self.weight_norm is None else float(self.weight_norm),
            "nan_inf_count": int(self.nan_inf_count),
            "safety_fallback_required": bool(self.safety_fallback_required),
            "failure_reasons": [str(reason) for reason in self.failure_reasons],
        }


@dataclass(frozen=True)
class SourceMaskSlcResult:
    """source-mask SLC の raw/effective 出力と診断量を保持する。

    このクラスは、固定整相後 beam output に対する source-mask SLC の結果を保持する。
    入力は `beam_output[beam, sample]` と source mask であり、出力は raw candidate、
    safety fallback 後の effective output、推定キャンセル成分、重み、参照 beam、健全性である。

    BL / FRAZ / BTR の metric 計算、source 検出、固定整相係数生成は責務に含めない。
    信号処理上は、A2 方式が実際に何を差し引いたかと、運用へ出す出力を分離する結果型に位置づく。
    """

    raw_output: NDArray[Any]
    effective_output: NDArray[Any]
    cancel_output: NDArray[Any]
    weights: NDArray[Any] | None
    source_reference_beams: NDArray[np.int64]
    non_source_beams: NDArray[np.int64]
    source_mask: NDArray[np.bool_]
    health: SourceMaskSlcHealth


class SourceMaskNonSourceLeakageSubtractor:
    """source beam reference から non-source beam の漏れ込みを差し引く SLC 実行器。

    このクラスは、固定整相後の `beam_output[beam, sample]` を入力として、
    source mask 内 beam を reference とし、non-source beam ごとに
    `y_b[n] = d_b[n] - eta * h_b^H x_S[n]` を適用する。

    入力は beam-domain 時間波形、source mask、任意の source reference beam index であり、
    出力は source beam copy-through と non-source 更新を含む全 beam 出力
    `[n_beam, n_sample]` である。

    source 検出、固定整相、周波数 bin 別 covariance solve、
    任意複素 channel 重み設計は責務に含めない。
    信号処理上は、リアルタイム経路へ重い ABF を入れず、固定整相後の表示 beam に対して
    source-correlated leakage だけを軽量に抑える後段 beam-domain SLC に位置づく。
    """

    def __init__(self, config: SourceMaskSlcConfig) -> None:
        self.config = config
        self.capacity_checker = SlcReferenceCapacityChecker(
            min_ref=int(config.min_ref),
            sample_per_dof=float(config.sample_per_dof),
            tap_len=int(config.tap_len),
        )
        self.solver = BlockLeastSquaresSlcSolver(loading=float(config.loading))

    def process(
        self,
        beam_output: NDArray[Any],
        source_sector_mask: SourceSectorMask,
        source_reference_beams: NDArray[Any] | None = None,
    ) -> SourceMaskSlcResult:
        """固定整相後 beam output へ source-mask SLC を適用する。

        Args:
            beam_output: 固定整相後の beam-domain 時間波形。shape は `[n_beam, n_sample]`。
                axis=0 が beam、axis=1 が時間 sample である。実数・複素を受け付ける。
            source_sector_mask: source / non-source mask。`source_mask` の shape は `[n_beam]`。
            source_reference_beams: source reference として使う beam index。
                `None` の場合は source 中心 beam と設定した近傍 beam から作る。shape は `[n_ref]`。

        Returns:
            raw candidate、effective output、cancel output、係数、診断量を含む結果。

        Raises:
            ValueError: 入力 shape、mask、reference beam が不正な場合。

        境界条件:
            source reference が空、non-source が空、sample 数が tap_len 未満、
            参照容量不足の条件では、
            SLC を無効化し fixed baseline を effective output として返す。
            source mask 内 beam は source として観測を維持するため、通常設定では raw/effective とも
            固定整相出力を copy-through する。
        """
        beam_signals = np.asarray(beam_output)
        require(beam_signals.ndim == 2, "beam_output must have shape (n_beam, n_sample).")
        require(
            beam_signals.shape[0] == source_sector_mask.source_mask.size,
            "beam_output and source mask must agree on n_beam.",
        )
        require(beam_signals.shape[1] > 0, "beam_output must contain at least one sample.")

        n_beam = int(beam_signals.shape[0])
        n_sample = int(beam_signals.shape[1])
        tap_len = int(self.config.tap_len)
        n_valid_sample = max(0, n_sample - tap_len + 1)
        output_dtype = np.result_type(beam_signals.dtype, np.complex128)
        raw_output = beam_signals.astype(output_dtype, copy=True)
        effective_output = beam_signals.astype(output_dtype, copy=True)
        cancel_output = np.zeros((n_beam, n_sample), dtype=output_dtype)

        reference_indices = self._resolve_source_reference_beams(
            source_sector_mask=source_sector_mask,
            source_reference_beams=source_reference_beams,
        )
        non_source_beams = np.flatnonzero(source_sector_mask.non_source_mask).astype(np.int64)
        capacity = self.capacity_checker.check(
            n_ref=int(reference_indices.size),
            block_size=int(n_valid_sample),
        )

        input_nan_inf_count = int(np.count_nonzero(~np.isfinite(beam_signals)))
        if input_nan_inf_count > 0:
            # 入力に NaN / inf がある場合は、共分散や最小二乗解へ進むと異常値を拡散する。
            # fixed baseline 自体の異常は上流問題として記録し、SLC は無効化する。
            return self._disabled_result(
                raw_output=raw_output,
                effective_output=effective_output,
                cancel_output=cancel_output,
                source_reference_beams=reference_indices,
                non_source_beams=non_source_beams,
                source_mask=source_sector_mask.source_mask,
                capacity=capacity,
                mode="DISABLED_NON_FINITE_INPUT",
                reasons=("non_finite_input",),
                nan_inf_count=input_nan_inf_count,
            )

        if non_source_beams.size == 0:
            # source mask が全 beam を覆う場合、抑圧対象の non-source sector がない。
            # 評価不能な状態で係数を解かず、固定整相をそのまま返す。
            return self._disabled_result(
                raw_output=raw_output,
                effective_output=effective_output,
                cancel_output=cancel_output,
                source_reference_beams=reference_indices,
                non_source_beams=non_source_beams,
                source_mask=source_sector_mask.source_mask,
                capacity=capacity,
                mode="DISABLED_NON_SOURCE_EMPTY",
                reasons=("non_source_empty",),
                nan_inf_count=0,
            )

        if not capacity.is_feasible:
            # reference 数または snapshot 数が足りない状態で source leakage 推定を行うと、
            # source 成分を別方位へ押し出す危険があるため、固定整相へ安全側に倒す。
            return self._disabled_result(
                raw_output=raw_output,
                effective_output=effective_output,
                cancel_output=cancel_output,
                source_reference_beams=reference_indices,
                non_source_beams=non_source_beams,
                source_mask=source_sector_mask.source_mask,
                capacity=capacity,
                mode="DISABLED_REFERENCE_CAPACITY",
                reasons=("reference_capacity_insufficient",),
                nan_inf_count=0,
            )

        source_reference_output = beam_signals[reference_indices, :]
        # Xs_tap shape: [n_ref * L, n_sample - L + 1]。
        # axis=0 は source reference beam と tap の結合自由度、
        # axis=1 は full tap が揃う時間 sample である。
        source_reference_tapped = build_time_tapped_reference_matrix(
            reference_output=source_reference_output,
            tap_len=tap_len,
        )
        aligned_non_source_output = beam_signals[non_source_beams, tap_len - 1 :]

        # R_ss = X_S X_S^H / K は source reference の自己共分散であり、
        # r_sd = X_S d_b^* / K は non-source beam b に漏れた
        # source-correlated 成分を推定する右辺である。
        covariance_matrix = np.asarray(
            (source_reference_tapped @ source_reference_tapped.conj().T) / float(n_valid_sample),
            dtype=output_dtype,
        )
        cross_correlations = np.zeros(
            (int(non_source_beams.size), int(source_reference_tapped.shape[0])),
            dtype=output_dtype,
        )
        for non_source_number in range(int(non_source_beams.size)):
            cross_correlations[non_source_number] = (
                source_reference_tapped @ aligned_non_source_output[non_source_number].conj()
            ) / float(n_valid_sample)

        condition_number = _loaded_covariance_condition_number(
            covariance_matrix,
            loading=float(self.config.loading),
        )
        try:
            weights = self.solver.solve(R=covariance_matrix, r=cross_correlations)
        except np.linalg.LinAlgError:
            return self._disabled_result(
                raw_output=raw_output,
                effective_output=effective_output,
                cancel_output=cancel_output,
                source_reference_beams=reference_indices,
                non_source_beams=non_source_beams,
                source_mask=source_sector_mask.source_mask,
                capacity=capacity,
                mode="SAFETY_FALLBACK",
                reasons=("linear_solve_failed",),
                nan_inf_count=0,
                condition_number=condition_number,
            )

        # C[b, n] = h_b^H X_S[n] = conj(W[b]) @ X_S[:, n]。
        # source mask 内 beam は観測対象として残すため、キャンセルは non-source beam にだけ入れる。
        valid_cancel_output = np.conj(weights) @ source_reference_tapped
        cancel_output[non_source_beams, tap_len - 1 :] = valid_cancel_output
        raw_output[non_source_beams, tap_len - 1 :] = (
            aligned_non_source_output - float(self.config.eta) * valid_cancel_output
        )
        if bool(self.config.copy_source_beams):
            raw_output[source_sector_mask.source_mask, :] = beam_signals[
                source_sector_mask.source_mask, :
            ]

        weight_norm = float(np.linalg.norm(np.asarray(weights, dtype=np.complex128)))
        raw_nan_inf_count = int(np.count_nonzero(~np.isfinite(raw_output)))
        failure_reasons: list[str] = []
        if raw_nan_inf_count > 0:
            failure_reasons.append("non_finite_raw_output")
        condition_number_exceeded = condition_number > float(self.config.condition_number_limit)
        if not bool(np.isfinite(condition_number)) or condition_number_exceeded:
            # condition_number_limit は資料の fail 条件 1e8 に合わせる。
            # 悪条件のまま出力を採用すると block 間の重み急変や過大キャンセルを起こす。
            failure_reasons.append("condition_number_limit_exceeded")
        if self.config.weight_norm_limit is not None and weight_norm > float(
            self.config.weight_norm_limit
        ):
            # weight norm が制限を超える場合、reference の微小差を過大に増幅している可能性がある。
            failure_reasons.append("weight_norm_limit_exceeded")

        safety_fallback_required = bool(len(failure_reasons) > 0)
        mode = "SAFETY_FALLBACK" if safety_fallback_required else "NORMAL"
        if safety_fallback_required:
            effective_output = beam_signals.astype(output_dtype, copy=True)
        else:
            effective_output = raw_output.copy()

        health = SourceMaskSlcHealth(
            mode=mode,
            capacity=capacity,
            eta=float(0.0 if safety_fallback_required else self.config.eta),
            loading=float(self.config.loading),
            tap_len=tap_len,
            n_ref=int(reference_indices.size),
            n_non_source=int(non_source_beams.size),
            condition_number=float(condition_number),
            weight_norm=weight_norm,
            nan_inf_count=int(raw_nan_inf_count),
            safety_fallback_required=safety_fallback_required,
            failure_reasons=tuple(failure_reasons),
        )
        return SourceMaskSlcResult(
            raw_output=raw_output,
            effective_output=effective_output,
            cancel_output=cancel_output,
            weights=weights,
            source_reference_beams=reference_indices.copy(),
            non_source_beams=non_source_beams.copy(),
            source_mask=source_sector_mask.source_mask.copy(),
            health=health,
        )

    def _resolve_source_reference_beams(
        self,
        *,
        source_sector_mask: SourceSectorMask,
        source_reference_beams: NDArray[Any] | None,
    ) -> NDArray[np.int64]:
        """source reference beam index を決定する。"""
        if source_reference_beams is not None:
            reference_indices = np.asarray(source_reference_beams, dtype=np.int64)
            require(reference_indices.ndim == 1, "source_reference_beams must have shape (n_ref,).")
        else:
            reference_candidates: list[int] = []
            n_beam = int(source_sector_mask.source_mask.size)
            for source_index in source_sector_mask.source_beam_indices.tolist():
                reference_guard = int(self.config.source_reference_guard_beam_count)
                start = max(0, int(source_index) - reference_guard)
                stop = min(n_beam, int(source_index) + reference_guard + 1)
                reference_candidates.extend(range(start, stop))
            reference_indices = np.asarray(sorted(set(reference_candidates)), dtype=np.int64)

        if reference_indices.size == 0:
            return reference_indices.astype(np.int64, copy=True)

        require(
            bool(
                np.all(
                    (0 <= reference_indices)
                    & (reference_indices < source_sector_mask.source_mask.size)
                )
            ),
            "source_reference_beams contain out-of-range index.",
        )
        require(
            bool(np.all(source_sector_mask.source_mask[reference_indices])),
            "source_reference_beams must be inside source mask.",
        )
        return np.unique(reference_indices).astype(np.int64, copy=False)

    def _disabled_result(
        self,
        *,
        raw_output: NDArray[Any],
        effective_output: NDArray[Any],
        cancel_output: NDArray[Any],
        source_reference_beams: NDArray[np.int64],
        non_source_beams: NDArray[np.int64],
        source_mask: NDArray[np.bool_],
        capacity: SlcReferenceCapacityDecision,
        mode: str,
        reasons: tuple[str, ...],
        nan_inf_count: int,
        condition_number: float | None = None,
    ) -> SourceMaskSlcResult:
        """SLC を無効化して fixed baseline を返す結果を作る。"""
        health = SourceMaskSlcHealth(
            mode=str(mode),
            capacity=capacity,
            eta=0.0,
            loading=float(self.config.loading),
            tap_len=int(self.config.tap_len),
            n_ref=int(source_reference_beams.size),
            n_non_source=int(non_source_beams.size),
            condition_number=condition_number,
            weight_norm=None,
            nan_inf_count=int(nan_inf_count),
            safety_fallback_required=True,
            failure_reasons=tuple(reasons),
        )
        return SourceMaskSlcResult(
            raw_output=raw_output.copy(),
            effective_output=effective_output.copy(),
            cancel_output=cancel_output.copy(),
            weights=None,
            source_reference_beams=source_reference_beams.copy(),
            non_source_beams=non_source_beams.copy(),
            source_mask=source_mask.copy(),
            health=health,
        )


def _relative_diagonal_loading_power(covariance_matrix: NDArray[Any], loading: float) -> float:
    """平均対角 power に対する相対対角 loading を実 power へ変換する。"""
    matrix = np.asarray(covariance_matrix)
    require(matrix.ndim == 2, "covariance_matrix must have shape (n_ref, n_ref).")
    require(matrix.shape[0] == matrix.shape[1], "covariance_matrix must be square.")
    require_non_negative_float("loading", float(loading))

    n_ref = int(matrix.shape[0])
    average_power = float(np.real(np.trace(matrix)) / float(n_ref))
    if not bool(np.isfinite(average_power)) or average_power <= 0.0:
        # source reference が無音に近い場合でも loading の行列を作れるよう、1.0 を基準にする。
        # ここで 0 を使うと loaded covariance も特異になり、fallback 判定前に solve が破綻する。
        average_power = 1.0
    return float(loading) * average_power


def _loaded_covariance_condition_number(covariance_matrix: NDArray[Any], loading: float) -> float:
    """対角 loading 後の source-reference 共分散条件数を返す。"""
    matrix = np.asarray(covariance_matrix)
    require(matrix.ndim == 2, "covariance_matrix must have shape (n_ref, n_ref).")
    require(matrix.shape[0] == matrix.shape[1], "covariance_matrix must be square.")
    loading_power = _relative_diagonal_loading_power(matrix, float(loading))
    # R_loaded = R_ss + λ mean(diag(R_ss)) I。
    # source reference 間が高相関でも、この条件数を summary に残すことで係数発散の兆候を評価できる。
    loaded = matrix + loading_power * np.eye(matrix.shape[0], dtype=matrix.dtype)
    return float(np.linalg.cond(loaded))


__all__ = [
    "SourceMaskNonSourceLeakageSubtractor",
    "SourceMaskSlcConfig",
    "SourceMaskSlcHealth",
    "SourceMaskSlcResult",
]
