"""方式3の完成etaから一周期遅延Weightで方位別共分散を選択・合成する。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .._validation import require, require_positive_float
from .covariance_snapshot_schedule import (
    CovarianceSnapshotCenterSchedule,
    DirectionMatchedCovarianceAccumulator,
)


class CovarianceFallbackSource(IntEnum):
    """周波数binごとの公開共分散sourceを表す。"""

    INVALID = 0
    METHOD3_WEIGHTED = 1
    METHOD2 = 2
    PREVIOUS_METHOD3 = 3
    DIAGONAL_SAFE = 4


class CovarianceFallbackReason(IntEnum):
    """周波数binで最終的に選ばれたfallback理由を表す。"""

    NONE = 0
    NO_PREVIOUS_WEIGHT = 1
    WEIGHT_CONDITION_FAILED = 2
    METHOD2_USED = 3
    PREVIOUS_METHOD3_USED = 4
    DIAGONAL_SAFE_USED = 5
    NO_VALID_OUTPUT = 6


@dataclass(frozen=True)
class DirectionCovarianceSelectionConfig:
    """soft Weightとfallback成立条件を保持する。

    Attributes:
        gamma_off: soft Weightが0となる周波数別eta。shapeは`[n_bin]`。
        gamma_on: soft Weightが1となる周波数別eta。shapeは`[n_bin]`。
        minimum_weight_sum: 方式3合成に必要なWeight総和。
        minimum_effective_direction_count: 方式3合成に必要な有効方位数。
        previous_hold_cycles: 前回方式3共分散を保持する2秒周期数。
        hermitian_tolerance: Hermitian相対誤差上限。
        psd_tolerance: trace基準の負固有値許容比。
        power_floor: traceと分母の数値下限。
    """

    gamma_off: NDArray[np.float32]
    gamma_on: NDArray[np.float32]
    minimum_weight_sum: float = 1.0
    minimum_effective_direction_count: float = 1.0
    previous_hold_cycles: int = 2
    hermitian_tolerance: float = 1.0e-5
    psd_tolerance: float = 1.0e-5
    power_floor: float = 1.0e-20


@dataclass(frozen=True)
class DirectionCovarianceSelectionResult:
    """2秒完成周期で公開するWeightとfallback後共分散を保持する。

    Attributes:
        completed_cycle_count: 完成周期数。
        eta: 完成eta。shapeは`[n_direction,n_bin]`。
        eta_valid: eta有効mask。shapeは`[n_direction,n_bin]`。
        active_channel_count: etaで使用したchannel数。shapeは`[n_bin]`。
        effective_channel_count: shadingを含む`N_eff`。shapeは`[n_bin]`。
        noise_eta_reference: 空間白色雑音の`1/N_eff`。shapeは`[n_bin]`。
        completed_weight: 今周期etaから作り次周期に使うWeight。shapeは`[n_direction,n_bin]`。
        applied_weight: 今周期共分散へ適用した前周期Weight。shapeは`[n_direction,n_bin]`。
        weighted_covariance: fallback後の公開候補。shapeは`[n_ch,n_ch,n_bin]`。
        covariance_valid: 公開可能bin。shapeは`[n_bin]`。
        fallback_source: `CovarianceFallbackSource`値。shapeは`[n_bin]`。
        fallback_reason: `CovarianceFallbackReason`値。shapeは`[n_bin]`。
        weight_sum: 適用Weight総和。shapeは`[n_bin]`。
        effective_direction_count: Weightの実効方位数。shapeは`[n_bin]`。
    """

    completed_cycle_count: int
    eta: NDArray[np.float32]
    eta_valid: NDArray[np.bool_]
    active_channel_count: NDArray[np.int32]
    effective_channel_count: NDArray[np.float32]
    noise_eta_reference: NDArray[np.float32]
    completed_weight: NDArray[np.float32]
    applied_weight: NDArray[np.float32]
    weighted_covariance: NDArray[np.complex64]
    covariance_valid: NDArray[np.bool_]
    fallback_source: NDArray[np.int8]
    fallback_reason: NDArray[np.int8]
    weight_sum: NDArray[np.float32]
    effective_direction_count: NDArray[np.float32]


class DirectionMatchedCovarianceSelector:
    """方式3の2秒完成状態だけから一周期遅延Weightと安全な共分散を公開する。

    1秒信号を入力し、2秒完成時だけ結果を返す。周期途中のetaやWeightは返さない。
    threshold校正、scene生成、方式比較、MVDR設計は責務に含めない。
    """

    def __init__(
        self,
        schedule: CovarianceSnapshotCenterSchedule,
        steering_table: NDArray[Any],
        config: DirectionCovarianceSelectionConfig,
        *,
        integration_time_seconds: float,
        channel_weight_table: NDArray[Any] | None = None,
    ) -> None:
        """固定steering、soft threshold、積分時間、周波数別shadingを設定する。

        Args:
            schedule: 2秒中心sample・方位一致表。
            steering_table: 物理steering。shapeは`[n_ch,n_bin,n_direction]`。
            config: 周波数別thresholdとfallback成立条件。
            integration_time_seconds: 方位別指数積分時間。単位はs。
            channel_weight_table: steering power用shading。shapeは`[n_ch,n_bin]`。
                `None`では全channelを係数1で使用する。

        Raises:
            ValueError: threshold、shading、shapeまたは成立条件が不正な場合。

        境界条件:
            shadingが0のchannelは該当binのetaへ寄与しないが、方位別共分散自体は
            fallbackやMVDRで使えるよう全channelのまま保持する。
        """

        self.schedule = schedule
        self.config = config
        gamma_off = np.asarray(config.gamma_off, dtype=np.float32)
        gamma_on = np.asarray(config.gamma_on, dtype=np.float32)
        n_bin = schedule.snapshot_length_samples // 2 + 1
        require(gamma_off.shape == (n_bin,), "gamma_off must have shape (n_bin,).")
        require(gamma_on.shape == (n_bin,), "gamma_on must have shape (n_bin,).")
        require(bool(np.all(np.isfinite(gamma_off))) and bool(np.all(np.isfinite(gamma_on))), "gamma tables must be finite.")
        require(bool(np.all(gamma_off < gamma_on)), "gamma_off must be lower than gamma_on for every bin.")
        require_positive_float("minimum_weight_sum", float(config.minimum_weight_sum))
        require_positive_float("minimum_effective_direction_count", float(config.minimum_effective_direction_count))
        require(int(config.previous_hold_cycles) >= 0, "previous_hold_cycles must be non-negative.")
        require_positive_float("hermitian_tolerance", float(config.hermitian_tolerance))
        require_positive_float("psd_tolerance", float(config.psd_tolerance))
        require_positive_float("power_floor", float(config.power_floor))
        self._gamma_off = gamma_off
        self._gamma_on = gamma_on
        self.accumulator = DirectionMatchedCovarianceAccumulator(
            schedule,
            integration_time_seconds=integration_time_seconds,
            steering_table=steering_table,
            channel_weight_table=channel_weight_table,
            eta_denominator_floor=config.power_floor,
        )
        n_direction = int(schedule.global_direction_azimuth_deg.size)
        self._published_weight: NDArray[np.float32] | None = None
        self._previous_method3 = np.zeros((schedule.n_ch, schedule.n_ch, n_bin), dtype=np.complex64)
        self._previous_valid = np.zeros(n_bin, dtype=np.bool_)
        self._previous_age = np.zeros(n_bin, dtype=np.int32)
        self._input_series_id: str | None = None
        self._zero_weight = np.zeros((n_direction, n_bin), dtype=np.float32)

    def reset(self) -> None:
        """周期途中、公開Weight、保持共分散を全て失効させる。"""

        self.accumulator.reset()
        self._published_weight = None
        self._previous_method3.fill(np.complex64(0.0 + 0.0j))
        self._previous_valid.fill(False)
        self._previous_age.fill(0)

    def process_one_second(
        self,
        signal: NDArray[Any],
        *,
        input_series_id: str,
        method2_covariance: NDArray[Any] | None = None,
    ) -> DirectionCovarianceSelectionResult | None:
        """1秒入力を更新し、2秒完成境界だけで選択結果を返す。

        Args:
            signal: 実信号。shapeは`[n_ch,fs]`。
            input_series_id: 入力系列識別子。変更時は旧状態を全破棄する。
            method2_covariance: fallback候補。shapeは`[n_ch,n_ch,n_bin]`。

        Returns:
            2秒周期途中は`None`、完成時は固定shapeの選択結果。
        """

        if self._input_series_id != input_series_id:
            self.reset()
            self._input_series_id = str(input_series_id)
        try:
            update = self.accumulator.process_one_second(signal)
        except Exception:
            # 例外後に途中周期を再利用すると異系列snapshotが混ざるため、安全側として全状態を破棄する。
            self.reset()
            raise
        if update.processed_second_count % 2 != 0:
            return None

        metrics = self.accumulator.completed_steering_metrics()
        completed_weight = self._soft_weight(metrics.eta, metrics.eta_valid)
        applied_weight = self._zero_weight.copy() if self._published_weight is None else self._published_weight.copy()
        result = self._compose_with_fallback(
            applied_weight,
            metrics.eta,
            metrics.eta_valid,
            metrics.active_channel_count,
            metrics.effective_channel_count,
            metrics.noise_eta_reference,
            completed_weight,
            metrics.completed_cycle_count,
            method2_covariance,
        )
        # eta_tから作ったWeightは合成完了後にpublishし、同じ周期R_tへ即時適用しない。
        self._published_weight = completed_weight.copy()
        return result

    def _soft_weight(self, eta: NDArray[np.float32], valid: NDArray[np.bool_]) -> NDArray[np.float32]:
        """周波数別gamma tableからsoft Weightを作る。"""

        denominator = self._gamma_on - self._gamma_off
        weight = np.clip(
            (eta - self._gamma_off[np.newaxis, :]) / denominator[np.newaxis, :],
            0.0,
            1.0,
        ).astype(np.float32)
        weight[~valid] = np.float32(0.0)
        return weight

    def _compose_with_fallback(
        self,
        applied_weight: NDArray[np.float32],
        eta: NDArray[np.float32],
        eta_valid: NDArray[np.bool_],
        active_channel_count: NDArray[np.int32],
        effective_channel_count: NDArray[np.float32],
        noise_eta_reference: NDArray[np.float32],
        completed_weight: NDArray[np.float32],
        cycle_count: int,
        method2_covariance: NDArray[Any] | None,
    ) -> DirectionCovarianceSelectionResult:
        """前周期Weightを適用しbinごとのfallback順序を実行する。"""

        n_bin = self.accumulator.n_bin
        output = np.zeros((self.schedule.n_ch, self.schedule.n_ch, n_bin), dtype=np.complex64)
        valid_output = np.zeros(n_bin, dtype=np.bool_)
        source = np.full(n_bin, int(CovarianceFallbackSource.INVALID), dtype=np.int8)
        initial_reason = (
            CovarianceFallbackReason.NO_PREVIOUS_WEIGHT
            if self._published_weight is None
            else CovarianceFallbackReason.WEIGHT_CONDITION_FAILED
        )
        reason = np.full(n_bin, int(initial_reason), dtype=np.int8)
        weight_sum = np.sum(applied_weight, axis=0, dtype=np.float32)
        weight_square_sum = np.sum(applied_weight**2, axis=0, dtype=np.float32)
        effective_count = np.zeros(n_bin, dtype=np.float32)
        nonzero_square = weight_square_sum > np.float32(self.config.power_floor)
        effective_count[nonzero_square] = weight_sum[nonzero_square] ** 2 / weight_square_sum[nonzero_square]

        weighted_numerator = np.einsum(
            "dk,dijk->ijk",
            applied_weight,
            self.accumulator.direction_covariance,
            optimize=True,
        )
        method2 = None if method2_covariance is None else np.asarray(method2_covariance, dtype=np.complex64)
        if method2 is not None:
            require(method2.shape == output.shape, "method2_covariance must have shape (n_ch, n_ch, n_bin).")
        self._previous_age[self._previous_valid] += 1

        diagonal_power = np.mean(
            np.real(np.diagonal(self.accumulator.direction_covariance, axis1=1, axis2=2)),
            axis=0,
        ).T
        for bin_index in range(n_bin):
            weight_ready = (
                float(weight_sum[bin_index]) >= float(self.config.minimum_weight_sum)
                and float(effective_count[bin_index]) >= float(self.config.minimum_effective_direction_count)
            )
            if weight_ready:
                candidate = weighted_numerator[:, :, bin_index] / weight_sum[bin_index]
                candidate = np.asarray(0.5 * (candidate + candidate.conj().T), dtype=np.complex64)
                if self._covariance_is_valid(candidate):
                    output[:, :, bin_index] = candidate
                    valid_output[bin_index] = True
                    source[bin_index] = int(CovarianceFallbackSource.METHOD3_WEIGHTED)
                    reason[bin_index] = int(CovarianceFallbackReason.NONE)
                    self._previous_method3[:, :, bin_index] = candidate
                    self._previous_valid[bin_index] = True
                    self._previous_age[bin_index] = 0
                    continue
            if method2 is not None and self._covariance_is_valid(method2[:, :, bin_index]):
                output[:, :, bin_index] = method2[:, :, bin_index]
                valid_output[bin_index] = True
                source[bin_index] = int(CovarianceFallbackSource.METHOD2)
                reason[bin_index] = int(CovarianceFallbackReason.METHOD2_USED)
                continue
            if self._previous_valid[bin_index] and self._previous_age[bin_index] <= int(self.config.previous_hold_cycles):
                output[:, :, bin_index] = self._previous_method3[:, :, bin_index]
                valid_output[bin_index] = True
                source[bin_index] = int(CovarianceFallbackSource.PREVIOUS_METHOD3)
                reason[bin_index] = int(CovarianceFallbackReason.PREVIOUS_METHOD3_USED)
                continue
            diagonal = np.asarray(diagonal_power[:, bin_index], dtype=np.float32)
            if bool(np.all(np.isfinite(diagonal))) and float(np.sum(diagonal)) > float(self.config.power_floor):
                output[:, :, bin_index] = np.diag(diagonal).astype(np.complex64)
                valid_output[bin_index] = True
                source[bin_index] = int(CovarianceFallbackSource.DIAGONAL_SAFE)
                reason[bin_index] = int(CovarianceFallbackReason.DIAGONAL_SAFE_USED)
            else:
                reason[bin_index] = int(CovarianceFallbackReason.NO_VALID_OUTPUT)

        return DirectionCovarianceSelectionResult(
            completed_cycle_count=cycle_count,
            eta=eta.copy(),
            eta_valid=eta_valid.copy(),
            active_channel_count=active_channel_count.copy(),
            effective_channel_count=effective_channel_count.copy(),
            noise_eta_reference=noise_eta_reference.copy(),
            completed_weight=completed_weight.copy(),
            applied_weight=applied_weight,
            weighted_covariance=output,
            covariance_valid=valid_output,
            fallback_source=source,
            fallback_reason=reason,
            weight_sum=weight_sum,
            effective_direction_count=effective_count,
        )

    def _covariance_is_valid(self, covariance: NDArray[np.complex64]) -> bool:
        """有限性、Hermitian誤差、trace、半正定値性を検査する。"""

        if not bool(np.all(np.isfinite(covariance))):
            return False
        norm = float(np.linalg.norm(covariance))
        hermitian_error = float(np.linalg.norm(covariance - covariance.conj().T)) / max(
            norm, float(self.config.power_floor)
        )
        if hermitian_error > float(self.config.hermitian_tolerance):
            return False
        hermitian = np.asarray(0.5 * (covariance + covariance.conj().T), dtype=np.complex64)
        trace = float(np.real(np.trace(hermitian)))
        if not np.isfinite(trace) or trace <= float(self.config.power_floor):
            return False
        minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(hermitian)))
        return minimum_eigenvalue >= -float(self.config.psd_tolerance) * max(trace, float(self.config.power_floor))

    @property
    def steering_state_bytes(self) -> int:
        """事前計算steeringと2個のpower積分状態の常駐byte数を返す。"""

        return self.accumulator.steering_state_bytes
