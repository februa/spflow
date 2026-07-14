"""ビームフォーミング後のbeam-domain sidelobe cancellationを実装する。"""

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
from ..level_conversion import LevelConverter, level_20log10_rms

_NORMALIZED_RMS_LEVEL_CONVERTER = LevelConverter.for_definition(
    level_20log10_rms(reference_rms=1.0, reference_label="normalized RMS")
)


@dataclass(frozen=True)
class SlcConfig:
    """ビーム領域 SLC の設定値を保持する。

    このクラスは、guard 幅、対角ロード量、忘却時定数、艦首変化感度、
    最小参照ビーム数、サンプル不足時の安全側パラメータを一体で保持する。

    入力は beam 単位の guard、本数ベースの最小参照数、秒単位の忘却時定数、
    角度単位の艦首変化スケールなどであり、出力は `BeamDomainSLC` や
    診断コードが参照する不変設定オブジェクトである。

    固定整相そのものの設計、target 選定ロジック、方位センサ異常検出は責務に含めない。
    信号処理上は、固定整相後のビーム領域サイドローブキャンセラの制御パラメータに位置づく。
    """

    guard: int
    loading: float
    memory_time_sec: float
    heading_scale_deg: float
    min_ref: int = 4
    sample_per_dof: float = 5.0
    tap_len: int = 1
    eta_normal: float = 1.0
    eta_limited: float = 0.5
    enable_heading_forgetting: bool = True
    enable_output_safety_gate: bool = True
    max_output_power_increase_db: float = 1.0
    max_output_power_drop_db: float = 6.0
    max_cancel_power_relative_db: float = 6.0

    def __post_init__(self) -> None:
        """SLC 設定値の範囲を検証する。"""
        require_non_negative_float("guard", float(self.guard))
        require_non_negative_float("loading", float(self.loading))
        require_positive_float("memory_time_sec", float(self.memory_time_sec))
        require_positive_float("heading_scale_deg", float(self.heading_scale_deg))
        require_positive_int("min_ref", int(self.min_ref))
        require_positive_float("sample_per_dof", float(self.sample_per_dof))
        require_positive_int("tap_len", int(self.tap_len))
        require(0.0 <= float(self.eta_normal) <= 1.0, "eta_normal must lie in [0.0, 1.0].")
        require(0.0 <= float(self.eta_limited) <= 1.0, "eta_limited must lie in [0.0, 1.0].")
        require_non_negative_float("max_output_power_increase_db", float(self.max_output_power_increase_db))
        require_non_negative_float("max_output_power_drop_db", float(self.max_output_power_drop_db))
        require_non_negative_float("max_cancel_power_relative_db", float(self.max_cancel_power_relative_db))


@dataclass(frozen=True)
class SlcReferenceCapacityDecision:
    """参照ビーム数とサンプル数から SLC 有効可否を表す。"""

    n_ref: int
    block_size: int
    tap_len: int
    dof: int
    has_enough_reference_beams: bool
    has_enough_samples: bool
    is_feasible: bool

    def as_dict(self) -> dict[str, int | bool]:
        """JSON 化しやすい辞書へ変換する。"""
        return {
            "n_ref": int(self.n_ref),
            "block_size": int(self.block_size),
            "tap_len": int(self.tap_len),
            "dof": int(self.dof),
            "has_enough_reference_beams": bool(self.has_enough_reference_beams),
            "has_enough_samples": bool(self.has_enough_samples),
            "is_feasible": bool(self.is_feasible),
        }


@dataclass(frozen=True)
class SlcOutputSafetyDecision:
    """SLC 出力を採用してよいかを表す安全判定を保持する。

    このクラスは、固定整相 target 出力、SLC 後出力、推定キャンセル成分の
    RMS パワーを比較し、自己消去や過大キャンセル推定を検出する。

    入力は target ごとの dB20 指標であり、出力は fallback 要否と理由である。
    SLC 係数の推定や固定整相そのものは責務に含めない。
    信号処理上は、適応処理が固定整相より悪化した場合に安全側へ倒す運用 gate に位置づく。
    """

    fallback_required: bool
    reasons: tuple[str, ...]
    target_input_power_db20: NDArray[np.float64]
    slc_output_power_db20: NDArray[np.float64]
    cancel_power_db20: NDArray[np.float64]
    output_delta_db: NDArray[np.float64]
    cancel_relative_db: NDArray[np.float64]

    def as_dict(self) -> dict[str, object]:
        """JSON 化しやすい辞書へ変換する。"""
        return {
            "fallback_required": bool(self.fallback_required),
            "reasons": [str(reason) for reason in self.reasons],
            "target_input_power_db20": [float(value) for value in self.target_input_power_db20.tolist()],
            "slc_output_power_db20": [float(value) for value in self.slc_output_power_db20.tolist()],
            "cancel_power_db20": [float(value) for value in self.cancel_power_db20.tolist()],
            "output_delta_db": [float(value) for value in self.output_delta_db.tolist()],
            "cancel_relative_db": [float(value) for value in self.cancel_relative_db.tolist()],
        }


def _row_rms_db20(signals: NDArray[Any]) -> NDArray[np.float64]:
    """行ごとの RMS レベルを dB20 で返す。

    Args:
        signals: 評価信号。shape は `[n_row, n_sample]`。axis=0 が評価対象、
            axis=1 が時間サンプルである。実数・複素のどちらも受け付ける。

    Returns:
        行ごとの RMS レベル。shape は `[n_row]`、単位は dB20。

    Raises:
        ValueError: 入力が 2 次元でない場合。
    """
    signal_array = np.asarray(signals)
    require(signal_array.ndim == 2, "signals must have shape (n_row, n_sample).")

    # SLC 係数は複素になり得るため、RMS は abs^2 で評価する。
    # dB 化ではゼロ入力を -inf にせず、判定用に有限値として扱う。
    rms = np.sqrt(np.mean(np.abs(signal_array) ** 2, axis=1))
    return np.asarray(
        _NORMALIZED_RMS_LEVEL_CONVERTER.output_rms_to_level(
            rms,
            floor_db=_NORMALIZED_RMS_LEVEL_CONVERTER.float64_tiny_level_db,
        ),
        dtype=np.float64,
    )


def evaluate_slc_output_safety(
    target_output: NDArray[Any],
    slc_output: NDArray[Any],
    cancel_output: NDArray[Any],
    config: SlcConfig,
) -> SlcOutputSafetyDecision:
    """SLC 出力を採用するか固定整相へ戻すかを判定する。

    Args:
        target_output: 固定整相 target 出力。shape は `[n_target, n_sample]`。
        slc_output: SLC 後 target 出力。shape は `[n_target, n_sample]`。
        cancel_output: 推定キャンセル成分。shape は `[n_target, n_sample]`。
        config: SLC 設定。安全判定閾値を含む。

    Returns:
        fallback 要否、理由、target ごとの dB 指標を含む安全判定。

    境界条件:
        safety gate が無効な場合は、指標だけ計算し `fallback_required=False` を返す。
        target 出力が極小の場合でも dB 指標は有限値へ丸め、NaN 判定で安全側に倒す。
    """
    fixed_power_db20 = _row_rms_db20(target_output)
    slc_power_db20 = _row_rms_db20(slc_output)
    cancel_power_db20 = _row_rms_db20(cancel_output)
    output_delta_db = slc_power_db20 - fixed_power_db20
    cancel_relative_db = cancel_power_db20 - fixed_power_db20

    reasons: list[str] = []
    if not bool(np.all(np.isfinite(output_delta_db))) or not bool(np.all(np.isfinite(cancel_relative_db))):
        # NaN / inf が出た時点で係数推定または入力が破綻している。
        # 適応出力を採用すると異常値が後段へ伝播するため、固定整相へ戻す。
        reasons.append("non_finite_safety_metric")

    if bool(config.enable_output_safety_gate):
        if bool(np.any(output_delta_db > float(config.max_output_power_increase_db))):
            # SLC 後パワーが固定整相より増える場合、キャンセルではなく不要成分注入の可能性が高い。
            reasons.append("output_power_increase")
        if bool(np.any(output_delta_db < -float(config.max_output_power_drop_db))):
            # 出力が大きく落ちる場合、target 自己消去の可能性が高い。
            # target absent が保証されない L=1 時間領域方式では、この条件を安全側 fallback とする。
            reasons.append("output_power_drop")
        if bool(np.any(cancel_relative_db > float(config.max_cancel_power_relative_db))):
            # 推定キャンセル成分が target 出力より過大な場合、参照相関で desired を説明している危険がある。
            reasons.append("cancel_power_too_large")

    return SlcOutputSafetyDecision(
        fallback_required=bool(len(reasons) > 0),
        reasons=tuple(reasons),
        target_input_power_db20=fixed_power_db20.astype(np.float64),
        slc_output_power_db20=slc_power_db20.astype(np.float64),
        cancel_power_db20=cancel_power_db20.astype(np.float64),
        output_delta_db=output_delta_db.astype(np.float64),
        cancel_relative_db=cancel_relative_db.astype(np.float64),
    )


@dataclass(frozen=True)
class SlcProcessResult:
    """1 ブロック分の SLC 処理結果を保持する。

    このクラスは、固定整相後 target 出力に対する SLC 適用結果と診断量を保持する。
    入力は `BeamDomainSLC.process()` 内で決定した target / reference beam と共分散推定結果であり、
    出力は SLC 後波形、推定キャンセル波形、重み、参照容量、安全判定である。

    SLC 係数の再計算、BL / FRAZ / BTR 描画、target 方位の選定は責務に含めない。
    信号処理上は、方式評価で raw SLC 候補と safety fallback 後の運用出力を分離して読むための
    ブロック結果に位置づく。
    """

    Y: NDArray[Any]
    C: NDArray[Any]
    W: NDArray[Any] | None
    reference_beams: NDArray[np.int64]
    protected_mask: NDArray[np.bool_]
    capacity: SlcReferenceCapacityDecision
    alpha: float | None
    eta: float
    mode: str
    safety: SlcOutputSafetyDecision | None = None
    reference_blocking_matrix: NDArray[np.complex128] | None = None
    covariance_condition_number: float | None = None


class BeamGuardSelector:
    """target beam と guard から保護領域と参照領域を作る。

    このクラスは、待ち受けビーム数と guard 幅を固定し、target beam 群から
    保護マスクと reference beam index を導出する。

    入力は target beam index 列 `[n_target]` であり、出力は保護マスク `[n_beam]` と
    参照 beam index 列 `[n_ref]` である。

    guard 幅の自動最適化や target 優先度判定は責務に含めない。
    信号処理上は、SLC で self-nulling を避けるための参照集合生成器に位置づく。
    """

    def __init__(self, n_beam: int, guard: int) -> None:
        require_positive_int("n_beam", n_beam)
        require_non_negative_float("guard", float(guard))
        self.n_beam = int(n_beam)
        self.guard = int(guard)

    def make_protected_mask(self, target_beams: NDArray[Any]) -> NDArray[np.bool_]:
        """target beam と guard から保護マスクを作る。

        Args:
            target_beams: target beam index。shape は `[n_target]`。
                各要素は `0 <= beam < n_beam` を満たす整数である。

        Returns:
            保護マスク。shape は `[n_beam]`。
            `True` は target 本体または guard 領域であり、参照に使わない。

        Raises:
            ValueError: beam index が範囲外の場合。
        """
        target_indices = np.asarray(target_beams, dtype=np.int64)
        require(target_indices.ndim == 1, "target_beams must be a 1-D array.")
        require(target_indices.size > 0, "target_beams must not be empty.")
        # np.all は np.bool_ を返すため、検証関数へ渡す前に Python bool へ明示変換する。
        # Pylance / Pyright 上も bool 条件として確定させ、範囲外 beam を早期に拒否する。
        require(bool(np.all((0 <= target_indices) & (target_indices < self.n_beam))), "target_beams contain out-of-range indices.")

        protected = np.zeros(self.n_beam, dtype=bool)
        for target_beam in target_indices:
            beam_start = max(0, int(target_beam) - self.guard)
            beam_stop = min(self.n_beam, int(target_beam) + self.guard + 1)

            # target 近傍を参照へ入れると、target 自身の mainlobe 成分を
            # 干渉として学習して自己消去する危険があるため、guard 全体を保護領域にする。
            protected[beam_start:beam_stop] = True
        return protected

    def make_reference_beams(self, target_beams: NDArray[Any]) -> NDArray[np.int64]:
        """保護領域外の参照 beam index 列を返す。

        Args:
            target_beams: target beam index。shape は `[n_target]`。

        Returns:
            参照 beam index。shape は `[n_ref]`。単位は beam index である。

        境界条件:
            target と guard が全 beam を覆う場合、空配列を返す。
            その後の `SlcReferenceCapacityChecker` が参照不足として SLC を無効化する。
        """
        protected = self.make_protected_mask(target_beams)
        return np.where(~protected)[0].astype(np.int64)


class SlcReferenceCapacityChecker:
    """参照ビーム数とサンプル数から SLC の安全な有効可否を判定する。"""

    def __init__(self, min_ref: int, sample_per_dof: float, tap_len: int = 1) -> None:
        require_positive_int("min_ref", min_ref)
        require_positive_float("sample_per_dof", sample_per_dof)
        require_positive_int("tap_len", tap_len)
        self.min_ref = int(min_ref)
        self.sample_per_dof = float(sample_per_dof)
        self.tap_len = int(tap_len)

    def check(self, n_ref: int, block_size: int) -> SlcReferenceCapacityDecision:
        """参照本数とブロック長から SLC の有効可否を返す。

        Args:
            n_ref: guard 外の reference beam 数。単位は beam 本数。
            block_size: 共分散推定に使える有効サンプル数。単位は sample。
                時間タップ付き SLC では `n_sample - tap_len + 1` である。

        Returns:
            SLC 有効可否。shape を持たない scalar 診断値の集合である。

        境界条件:
            `block_size=0` は、tap を作るだけのサンプルがない状態を表す。
            この場合は例外ではなく `has_enough_samples=False` として安全側に無効化する。
        """
        require_non_negative_float("n_ref", float(n_ref))
        require_non_negative_float("block_size", float(block_size))

        dof = int(n_ref) * self.tap_len
        has_enough_reference_beams = int(n_ref) >= self.min_ref
        has_enough_samples = float(block_size) >= self.sample_per_dof * float(dof)
        return SlcReferenceCapacityDecision(
            n_ref=int(n_ref),
            block_size=int(block_size),
            tap_len=int(self.tap_len),
            dof=int(dof),
            has_enough_reference_beams=bool(has_enough_reference_beams),
            has_enough_samples=bool(has_enough_samples),
            is_feasible=bool(has_enough_reference_beams and has_enough_samples),
        )


def build_time_tapped_reference_matrix(reference_output: NDArray[Any], tap_len: int) -> NDArray[Any]:
    """reference beam 出力を時間タップ付き SLC 用の行列へ展開する。

    Args:
        reference_output: reference beam 信号。shape は `[n_ref, n_sample]`。
            axis=0 が reference beam、axis=1 が時間サンプルである。
        tap_len: 時間タップ数 `L`。単位は sample。

    Returns:
        時間タップ付き reference 行列。shape は `[n_ref * L, n_sample - L + 1]`。
        row は `lag=0, 1, ..., L-1` の順に reference beam 軸を積む。

    Raises:
        ValueError: `reference_output` が 2 次元でない、`tap_len` が正でない、
            または full tap を作るだけのサンプル数がない場合。

    境界条件:
        先頭 `L-1` サンプルは過去 reference が不足するため、ここでは出力しない。
        呼び出し側は SLC 出力の先頭 `L-1` サンプルを固定整相出力で埋める。
    """
    reference_signals = np.asarray(reference_output)
    require(reference_signals.ndim == 2, "reference_output must have shape (n_ref, n_sample).")
    require_positive_int("tap_len", int(tap_len))

    n_ref = int(reference_signals.shape[0])
    n_sample = int(reference_signals.shape[1])
    require(n_sample >= int(tap_len), "reference_output must contain at least tap_len samples.")

    n_valid_sample = n_sample - int(tap_len) + 1
    tapped = np.zeros((n_ref * int(tap_len), n_valid_sample), dtype=reference_signals.dtype)
    for lag_index in range(int(tap_len)):
        row_start = lag_index * n_ref
        row_stop = row_start + n_ref
        sample_start = int(tap_len) - 1 - lag_index
        sample_stop = sample_start + n_valid_sample

        # tapped[lag, ref, n] = u_ref[n + L - 1 - lag]。
        # lag=0 は現在サンプル、lag>0 は過去サンプルに対応する。
        # この並びにより W[target, lag*n_ref:(lag+1)*n_ref] が FIR 型キャンセラの tap になる。
        tapped[row_start:row_stop, :] = reference_signals[:, sample_start:sample_stop]
    return tapped


class HeadingAwareForgettingController:
    """艦首変化を考慮した忘却係数を計算する。"""

    def __init__(
        self,
        fs_hz: float,
        block_size: int,
        memory_time_sec: float,
        heading_scale_deg: float,
        alpha_min: float = 0.0,
        alpha_max: float = 0.9999,
    ) -> None:
        require_positive_float("fs_hz", fs_hz)
        require_positive_int("block_size", block_size)
        require_positive_float("memory_time_sec", memory_time_sec)
        require_positive_float("heading_scale_deg", heading_scale_deg)
        require(0.0 <= float(alpha_min) <= float(alpha_max) <= 1.0, "alpha_min and alpha_max must satisfy 0 <= alpha_min <= alpha_max <= 1.")
        self.fs_hz = float(fs_hz)
        self.block_size = int(block_size)
        self.memory_time_sec = float(memory_time_sec)
        self.heading_scale_deg = float(heading_scale_deg)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)

    @staticmethod
    def wrap_angle_deg(angle_deg: float) -> float:
        """角度差を `[-180, 180)` へ折り返す。"""
        return float((float(angle_deg) + 180.0) % 360.0 - 180.0)

    def compute(self, heading_deg: float, prev_heading_deg: float, heading_valid: bool) -> float:
        """時間忘却と艦首変化忘却を掛け合わせた `alpha` を返す。"""
        block_time_sec = float(self.block_size) / float(self.fs_hz)
        alpha_time = np.exp(-block_time_sec / float(self.memory_time_sec))

        if not bool(heading_valid):
            return float(np.clip(alpha_time, self.alpha_min, self.alpha_max))

        delta_heading_deg = abs(self.wrap_angle_deg(float(heading_deg) - float(prev_heading_deg)))
        alpha_heading = np.exp(-delta_heading_deg / float(self.heading_scale_deg))

        # 艦首変化が大きいほど、相対方位上の source 配置が過去ブロックと乖離する。
        # そのため、時間忘却だけでなく heading 忘却も掛け合わせて過去統計を強く減衰させる。
        alpha = alpha_time * alpha_heading
        return float(np.clip(alpha, self.alpha_min, self.alpha_max))


class SlcCovarianceEstimator:
    """時間領域 SLC の参照共分散と target-参照相互相関を忘却平均する。

    このクラスは、固定整相後 `beam_output[beam, sample]` から切り出した
    reference beam と target beam の時間サンプル列を入力として、周波数で分割しない
    1 組の `R_uu[n_dof, n_dof]` と `r_ud[n_target, n_dof]` を更新する。

    時間タップ付き SLC では、呼び出し側が reference を `[n_ref * L, n_valid_sample]` へ
    展開してから渡す。STFT、帯域別共分散、周波数ビン別 GSC は責務に含めない。
    信号処理上は、時間領域 beam-domain SLC の統計推定部に位置づく。
    """

    def __init__(self, n_ref: int, n_target: int, dtype: np.dtype | type[np.generic] = np.complex128) -> None:
        require_positive_int("n_ref", n_ref)
        require_positive_int("n_target", n_target)
        matrix_dtype = np.dtype(dtype)
        self.R = np.zeros((int(n_ref), int(n_ref)), dtype=matrix_dtype)
        self.r = np.zeros((int(n_target), int(n_ref)), dtype=matrix_dtype)
        self._initialized = False

    def update(self, U: NDArray[Any], D: NDArray[Any], alpha: float) -> tuple[NDArray[Any], NDArray[Any]]:
        """参照信号と target 信号から共分散を更新する。

        Args:
            U: reference beam 信号。shape は `[n_ref, n_sample]`。
            D: target beam 信号。shape は `[n_target, n_sample]`。
            alpha: 忘却係数。`0 <= alpha <= 1`。

        Returns:
            `(R, r)` を返す。
            `R` の shape は `[n_ref, n_ref]`、`r` の shape は `[n_target, n_ref]`。
        """
        reference_signals = np.asarray(U)
        target_signals = np.asarray(D)
        require(reference_signals.ndim == 2, "U must have shape (n_ref, n_sample).")
        require(target_signals.ndim == 2, "D must have shape (n_target, n_sample).")
        require(reference_signals.shape[1] == target_signals.shape[1], "U and D must agree on n_sample.")
        require(reference_signals.shape[0] == self.R.shape[0], "U and estimator must agree on n_ref.")
        require(target_signals.shape[0] == self.r.shape[0], "D and estimator must agree on n_target.")
        require(0.0 <= float(alpha) <= 1.0, "alpha must lie in [0.0, 1.0].")

        n_sample = reference_signals.shape[1]
        require(n_sample > 0, "U and D must contain at least one sample.")

        # R_hat[ref_i, ref_j] = (1 / N) Σ_n U[ref_i, n] conj(U[ref_j, n])。
        # 参照 beam 間の相関をここで集約し、後段の最小二乗 SLC 係数計算へ渡す。
        R_hat = (reference_signals @ reference_signals.conj().T) / float(n_sample)

        # r_hat[target, ref] = (1 / N) Σ_n U[ref, n] conj(D[target, n])。
        # target ごとに参照群で説明できる成分を推定する右辺ベクトルに対応する。
        r_hat = np.zeros_like(self.r)
        for target_index in range(target_signals.shape[0]):
            r_hat[target_index] = (reference_signals @ target_signals[target_index].conj()) / float(n_sample)

        if not self._initialized:
            # 初回更新では過去統計が存在しないため、R と r を current block の統計量で初期化する。
            # ここで alpha を掛けると、ゼロ初期値に対して不要なスケール縮退が入り loading 相対値が崩れる。
            self.R[...] = R_hat
            self.r[...] = r_hat
            self._initialized = True
        else:
            self.R[...] = float(alpha) * self.R + (1.0 - float(alpha)) * R_hat
            self.r[...] = float(alpha) * self.r + (1.0 - float(alpha)) * r_hat
        return self.R.copy(), self.r.copy()

    def reset(self) -> None:
        """内部統計をゼロへ戻す。"""
        self.R[...] = 0
        self.r[...] = 0
        self._initialized = False


def _relative_diagonal_loading_power(R: NDArray[Any], loading_ratio: float) -> float:
    """共分散の平均対角 power から対角ロードの実 power を計算する。

    Args:
        R: 参照共分散行列。shape は `[n_ref, n_ref]`。
        loading_ratio: 平均対角 power に対する対角ロード比。無次元。

    Returns:
        `R + loading_power I` に使う `loading_power`。単位は参照信号 power と同じである。

    境界条件:
        入力レベルが変わっても正則化の相対強度を保つため、SLC でも MVDR / LCMV と同じく
        絶対値ではなく平均対角 power に対する比で loading を解釈する。無音や blocking 後の
        極小 power では平均対角が 0 になり得るため、その場合は 1.0 を基準にして数値特異を避ける。
    """
    covariance_matrix = np.asarray(R)
    require(covariance_matrix.ndim == 2, "R must have shape (n_ref, n_ref).")
    require(covariance_matrix.shape[0] == covariance_matrix.shape[1], "R must be square.")
    require_non_negative_float("loading_ratio", float(loading_ratio))

    n_ref = int(covariance_matrix.shape[0])
    average_power = float(np.real(np.trace(covariance_matrix)) / float(n_ref))
    return float(loading_ratio) * (average_power if average_power > 0.0 else 1.0)


class BlockLeastSquaresSlcSolver:
    """対角ロード付き block least-squares により SLC 係数を求める。

    このクラスは、参照共分散 `R_uu` と target-reference 相互相関 `r_ud` から、
    target beam ごとの SLC 重み `W` を計算する。

    入力は共分散行列 `[n_ref, n_ref]` と相互相関 `[n_target, n_ref]`、
    出力は重み行列 `[n_target, n_ref]` である。

    共分散推定、reference beam 選定、SLC 出力波形の生成は責務に含めない。
    信号処理上は、`(R_uu + λ mean(diag(R_uu)) I)w = r_ud` を解く適応キャンセラ係数計算部に位置づく。
    """

    def __init__(self, loading: float) -> None:
        require_non_negative_float("loading", loading)
        self.loading = float(loading)

    def solve(self, R: NDArray[Any], r: NDArray[Any]) -> NDArray[Any]:
        """`(R + λ mean(diag(R)) I) w = r` を target ごとに解く。

        Args:
            R: 参照共分散行列。shape は `[n_ref, n_ref]`。
                axis=0/1 は reference beam 軸で、値の単位は入力信号 power に対応する。
            r: target-reference 相互相関。shape は `[n_target, n_ref]`。
                axis=0 が target beam、axis=1 が reference beam である。

        Returns:
            SLC 重み。shape は `[n_target, n_ref]`。
            `conj(W[target]) @ U` が target ごとの推定キャンセル成分になる。

        Raises:
            ValueError: `R` と `r` の次元、または reference beam 軸が整合しない場合。
            numpy.linalg.LinAlgError: 対角ロード後の行列が数値的に解けない場合。

        境界条件:
            対角ロード比 `λ` は平均対角 power に掛けてから加える。
            参照 beam 間相関が高い条件で `R_uu` が悪条件になり、SLC 重みが過大化することを抑える。
        """
        covariance_matrix = np.asarray(R)
        cross_correlations = np.asarray(r)
        require(covariance_matrix.ndim == 2, "R must have shape (n_ref, n_ref).")
        require(cross_correlations.ndim == 2, "r must have shape (n_target, n_ref).")
        require(covariance_matrix.shape[0] == covariance_matrix.shape[1], "R must be square.")
        require(cross_correlations.shape[1] == covariance_matrix.shape[0], "r and R must agree on n_ref.")

        n_ref = int(covariance_matrix.shape[0])
        loading_power = _relative_diagonal_loading_power(covariance_matrix, self.loading)
        loaded_covariance = covariance_matrix + loading_power * np.eye(n_ref, dtype=covariance_matrix.dtype)
        weights = np.zeros((cross_correlations.shape[0], n_ref), dtype=loaded_covariance.dtype)
        for target_index in range(cross_correlations.shape[0]):
            weights[target_index] = np.linalg.solve(loaded_covariance, cross_correlations[target_index])
        return weights


def _loaded_covariance_condition_number(R: NDArray[Any], loading: float) -> float:
    """対角ロード後の共分散行列の条件数を返す。

    Args:
        R: 参照共分散行列。shape は `[n_ref, n_ref]`。
        loading: 平均対角 power に対する対角ロード比。無次元。

    Returns:
        `R + loading * mean(diag(R)) I` の 2-norm 条件数。無次元比である。

    Raises:
        ValueError: `R` が正方行列でない場合。

    境界条件:
        SLC 係数を実際に解く行列は raw `R` ではなく、平均対角 power 基準の loaded covariance である。
        desired blocking 後に raw `R` が rank 落ちしても、loading 後の条件数を記録することで
        solver が見ている数値安定性を評価できる。
    """
    covariance_matrix = np.asarray(R)
    require(covariance_matrix.ndim == 2, "R must have shape (n_ref, n_ref).")
    require(covariance_matrix.shape[0] == covariance_matrix.shape[1], "R must be square.")

    n_ref = int(covariance_matrix.shape[0])
    loading_power = _relative_diagonal_loading_power(covariance_matrix, float(loading))
    loaded_covariance = covariance_matrix + loading_power * np.eye(n_ref, dtype=covariance_matrix.dtype)
    return float(np.linalg.cond(loaded_covariance))


def build_reference_blocking_matrix(desired_reference_response: NDArray[Any]) -> NDArray[np.complex128]:
    """reference 空間から protected target 応答を射影除去する blocking 行列を作る。

    Args:
        desired_reference_response: reference beam 上の desired 応答。
            shape は `[n_ref, n_constraint]`。axis=0 が reference beam、
            axis=1 が保護する target / constraint である。

    Returns:
        blocking 行列。shape は `[n_ref, n_ref]`。
        `U_blocked = B @ U` として reference beam 出力へ適用する。

    Raises:
        ValueError: 入力 shape が不正、または constraint が空の場合。

    Notes:
        `B = I - A (A^H A)^+ A^H` を使う。
        ここで `A` は reference beam 上の desired 応答である。
        reference に desired sidelobe が残ったまま SLC 共分散を作ると、
        target-only 条件でも desired を干渉として学習して自己消去する。
        そのため beam-domain GSC では、共分散推定前に desired 応答部分空間を blocking する。
    """
    response = np.asarray(desired_reference_response, dtype=np.complex128)
    require(response.ndim == 2, "desired_reference_response must have shape (n_ref, n_constraint).")
    require(response.shape[0] > 0, "desired_reference_response must contain at least one reference beam.")
    require(response.shape[1] > 0, "desired_reference_response must contain at least one constraint.")
    require(bool(np.all(np.isfinite(response))), "desired_reference_response must contain finite values.")

    gram = response.conj().T @ response
    # pinv を使うのは、複数 target や対称配置で constraint が線形従属に近くなる場合があるためである。
    # 特異な Gram で solve すると異常停止するが、blocking としては最小ノルム射影で安全に扱える。
    gram_inverse = np.linalg.pinv(gram)
    identity = np.eye(response.shape[0], dtype=np.complex128)
    return identity - response @ gram_inverse @ response.conj().T


class BeamDomainSLC:
    """固定整相後ビームへ時間領域 block least-squares 型の SLC を適用する。

    このクラスは、target beam と reference beam を分離し、参照容量判定、
    忘却係数計算、共分散更新、SLC 係数計算、干渉推定、フォールバックを統括する。

    入力は固定整相後ビーム出力 `[n_beam, n_sample]`、target beam index `[n_target]`、
    必要に応じて艦首方位と方位センサ正常/異常フラグであり、出力は SLC 後 beam 出力
    `[n_target, n_sample]` と干渉推定 `[n_target, n_sample]`、係数 `[n_target, n_ref * L]` である。

    固定整相そのものの計算、target beam をどう選ぶか、guard 幅の自動最適化は責務に含めない。
    信号処理上は、固定整相出力に対する時間領域 FIR 型 beam-domain SLC 実行器に位置づく。
    周波数ごとの共分散を持つ beam-domain GSC は、広帯域性能が不足した場合の拡張方式として扱う。
    """

    def __init__(self, n_beam: int, fs_hz: float, block_size: int, config: SlcConfig) -> None:
        require_positive_int("n_beam", n_beam)
        require_positive_float("fs_hz", fs_hz)
        require_positive_int("block_size", block_size)
        self.n_beam = int(n_beam)
        self.fs_hz = float(fs_hz)
        self.block_size = int(block_size)
        self.config = config

        self.guard_selector = BeamGuardSelector(n_beam=int(n_beam), guard=int(config.guard))
        self.capacity_checker = SlcReferenceCapacityChecker(
            min_ref=int(config.min_ref),
            sample_per_dof=float(config.sample_per_dof),
            tap_len=int(config.tap_len),
        )
        self.forgetting_controller = HeadingAwareForgettingController(
            fs_hz=float(fs_hz),
            block_size=int(block_size),
            memory_time_sec=float(config.memory_time_sec),
            heading_scale_deg=float(config.heading_scale_deg),
        )
        self.solver = BlockLeastSquaresSlcSolver(loading=float(config.loading))
        self.estimator: SlcCovarianceEstimator | None = None
        self.prev_heading_deg: float | None = None

    def reset(self) -> None:
        """忘却統計と前回艦首方位を破棄する。"""
        self.estimator = None
        self.prev_heading_deg = None

    def process(
        self,
        beam_output: NDArray[Any],
        target_beams: NDArray[Any],
        heading_deg: float | None = None,
        heading_valid: bool = False,
        desired_response_matrix: NDArray[Any] | None = None,
    ) -> SlcProcessResult:
        """1 ブロックの固定整相出力へ SLC を適用する。

        Args:
            beam_output: 固定整相後ビーム出力。shape は `[n_beam, n_sample]`。
                axis=0 がビーム、axis=1 が時間サンプルである。
            target_beams: target beam index。shape は `[n_target]`。
            heading_deg: 現ブロックの艦首方位。単位は deg。
            heading_valid: 方位センサが正常で艦首変化忘却を使ってよい場合は `True`。
            desired_response_matrix: protected target の beam-domain 応答。
                shape は `[n_beam, n_constraint]`。axis=0 が beam、axis=1 が target / constraint である。
                指定した場合、reference beam 出力へ blocking projector を適用してから共分散を作る。

        Returns:
            SLC 後出力、干渉推定、重み、reference beam、容量判定などを含む結果。
        """
        beam_signals = np.asarray(beam_output)
        require(beam_signals.ndim == 2, "beam_output must have shape (n_beam, n_sample).")
        require(beam_signals.shape[0] == self.n_beam, "beam_output and BeamDomainSLC must agree on n_beam.")
        require(beam_signals.shape[1] > 0, "beam_output must contain at least one sample.")

        target_indices = np.asarray(target_beams, dtype=np.int64)
        require(target_indices.ndim == 1, "target_beams must be a 1-D array.")
        require(target_indices.size > 0, "target_beams must not be empty.")
        # np.all は np.bool_ を返すため、検証関数へ渡す前に Python bool へ明示変換する。
        # target beam が範囲外の場合は、誤った guard / reference 選定へ進む前に停止する。
        require(bool(np.all((0 <= target_indices) & (target_indices < self.n_beam))), "target_beams contain out-of-range indices.")

        protected_mask = self.guard_selector.make_protected_mask(target_indices)
        reference_beams = self.guard_selector.make_reference_beams(target_indices)
        tap_len = int(self.config.tap_len)
        n_valid_sample = int(beam_signals.shape[1]) - tap_len + 1
        capacity_block_size = max(0, n_valid_sample)
        capacity = self.capacity_checker.check(n_ref=int(reference_beams.size), block_size=int(capacity_block_size))
        target_output = beam_signals[target_indices, :]

        if not capacity.is_feasible:
            # 参照不足状態で無理に SLC を動かすと、target を守る guard を削るか、
            # 悪条件な最小二乗解を使うことになり自己消去の危険が高い。
            # そのため安全側として固定整相出力をそのまま返す。
            return SlcProcessResult(
                Y=target_output.copy(),
                C=np.zeros_like(target_output),
                W=None,
                reference_beams=reference_beams.copy(),
                protected_mask=protected_mask.copy(),
                capacity=capacity,
                alpha=None,
                eta=0.0,
                mode="DISABLED",
            )

        reference_output = beam_signals[reference_beams, :]
        reference_blocking_matrix: NDArray[np.complex128] | None = None
        if desired_response_matrix is not None:
            response_matrix = np.asarray(desired_response_matrix, dtype=np.complex128)
            require(response_matrix.ndim == 2, "desired_response_matrix must have shape (n_beam, n_constraint).")
            require(response_matrix.shape[0] == self.n_beam, "desired_response_matrix and BeamDomainSLC must agree on n_beam.")
            desired_reference_response = response_matrix[reference_beams, :]
            reference_blocking_matrix = build_reference_blocking_matrix(desired_reference_response)

            # U_blocked shape: [n_ref, n_sample]。
            # axis=0 は reference beam、axis=1 は時間サンプルである。
            # desired 応答部分空間を射影除去してから R_uu を作り、target-only 条件での自己消去を防ぐ。
            reference_output = reference_blocking_matrix @ reference_output

        # U_tap shape: [n_ref * L, n_sample - L + 1]。
        # axis=0 は reference beam と tap の結合自由度、axis=1 は full tap が揃う時間サンプルである。
        # target 側も同じ有効サンプルへ揃え、FIR 型 SLC の式 y[n] = d[n] - Σ_l w_l^H u[n-l] に対応させる。
        tapped_reference_output = build_time_tapped_reference_matrix(reference_output=reference_output, tap_len=tap_len)
        aligned_target_output = target_output[:, tap_len - 1 :]

        if (
            self.estimator is None
            or self.estimator.R.shape[0] != tapped_reference_output.shape[0]
            or self.estimator.r.shape[0] != aligned_target_output.shape[0]
        ):
            self.estimator = SlcCovarianceEstimator(
                n_ref=int(tapped_reference_output.shape[0]),
                n_target=int(aligned_target_output.shape[0]),
                dtype=np.result_type(tapped_reference_output.dtype, aligned_target_output.dtype, np.complex128),
            )

        alpha = self._resolve_forgetting_factor(heading_deg=heading_deg, heading_valid=heading_valid)
        # 時間領域方式では周波数ごとに共分散を分けず、時間サンプル axis=1 全体から
        # R_uu[n_ref * L, n_ref * L] と r_ud[n_target, n_ref * L] を 1 組だけ推定する。
        # L>1 では短い FIR キャンセラになり、reference から target への相対的な時間ずれを吸収しやすくする。
        covariance_matrix, cross_correlations = self.estimator.update(
            U=tapped_reference_output,
            D=aligned_target_output,
            alpha=float(alpha),
        )
        covariance_condition_number = _loaded_covariance_condition_number(
            R=covariance_matrix,
            loading=float(self.config.loading),
        )
        weights = self.solver.solve(R=covariance_matrix, r=cross_correlations)

        # C_valid[target, n] = Σ_dof conj(W[target, dof]) U_tap[dof, n] により、
        # reference beam 群とその過去 tap で説明できる干渉成分だけを target ごとに推定する。
        valid_cancel_output = np.conj(weights) @ tapped_reference_output
        cancel_output = np.zeros_like(target_output, dtype=np.result_type(valid_cancel_output.dtype, target_output.dtype))
        cancel_output[:, tap_len - 1 :] = valid_cancel_output

        eta = float(self.config.eta_normal)
        raw_slc_output = target_output.astype(cancel_output.dtype, copy=True)
        raw_slc_output[:, tap_len - 1 :] = aligned_target_output - eta * valid_cancel_output
        safety = evaluate_slc_output_safety(
            target_output=target_output,
            slc_output=raw_slc_output,
            cancel_output=cancel_output,
            config=self.config,
        )

        mode = "NORMAL"
        slc_output = raw_slc_output
        effective_eta = eta
        if safety.fallback_required:
            # SLC が固定整相より悪化する兆候を検出した場合は、係数と推定成分は診断用に返しつつ、
            # 実際の出力 Y は固定整相へ戻す。これにより target 自己消去や不要成分注入を後段へ伝播させない。
            slc_output = target_output.copy()
            effective_eta = 0.0
            mode = "SAFETY_FALLBACK"
        elif heading_deg is not None and not bool(heading_valid):
            mode = "DEGRADED_HEADING"

        if heading_deg is not None:
            self.prev_heading_deg = float(heading_deg)

        return SlcProcessResult(
            Y=slc_output,
            C=cancel_output,
            W=weights,
            reference_beams=reference_beams.copy(),
            protected_mask=protected_mask.copy(),
            capacity=capacity,
            alpha=float(alpha),
            eta=effective_eta,
            mode=mode,
            safety=safety,
            reference_blocking_matrix=None if reference_blocking_matrix is None else reference_blocking_matrix.copy(),
            covariance_condition_number=float(covariance_condition_number),
        )

    def _resolve_forgetting_factor(self, heading_deg: float | None, heading_valid: bool) -> float:
        """現在ブロックで使う忘却係数を返す。"""
        block_time_sec = float(self.block_size) / float(self.fs_hz)
        alpha_time = float(np.exp(-block_time_sec / float(self.config.memory_time_sec)))

        if not bool(self.config.enable_heading_forgetting):
            return alpha_time
        if self.prev_heading_deg is None or heading_deg is None:
            return alpha_time
        return self.forgetting_controller.compute(
            heading_deg=float(heading_deg),
            prev_heading_deg=float(self.prev_heading_deg),
            heading_valid=bool(heading_valid),
        )


__all__ = [
    "SlcConfig",
    "SlcReferenceCapacityDecision",
    "SlcOutputSafetyDecision",
    "SlcProcessResult",
    "evaluate_slc_output_safety",
    "build_time_tapped_reference_matrix",
    "build_reference_blocking_matrix",
    "BeamGuardSelector",
    "SlcReferenceCapacityChecker",
    "HeadingAwareForgettingController",
    "SlcCovarianceEstimator",
    "BlockLeastSquaresSlcSolver",
    "BeamDomainSLC",
]
