"""方位別共分散から非対角の正規化空間相関統計を計算する。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .._validation import require, require_positive_float


FloatArray = NDArray[np.floating[Any]]


@dataclass(frozen=True)
class SpatialCorrelationStatistics:
    """非対角チャネルペアの正規化相関統計を保持する。

    このクラスは、方位別・周波数別の最大値、平均、中央値、95 percentileと、
    等間隔ULAを想定した基線index別平均を固定shapeで返す。

    入力信号の生成、共分散の積分、画像化、閾値による採否判定は責務に含めない。
    信号処理上は、方式3で生成した共分散の方位選択性を比較する観測量に位置づく。

    Attributes:
        maximum: 全非対角pairの最大値。shapeは`[n_direction,n_bin]`。
        mean: 全非対角pairの平均。shapeは`[n_direction,n_bin]`。
        median: 全非対角pairの中央値。shapeは`[n_direction,n_bin]`。
        percentile_95: 全非対角pairの95 percentile。shapeは`[n_direction,n_bin]`。
        baseline_mean: `abs(i-j)`別平均。shapeは`[n_direction,n_bin,n_ch-1]`。
        baseline_index: 基線index。shapeは`[n_ch-1]`、値は1から`n_ch-1`。
    """

    maximum: NDArray[np.float32]
    mean: NDArray[np.float32]
    median: NDArray[np.float32]
    percentile_95: NDArray[np.float32]
    baseline_mean: NDArray[np.float32]
    baseline_index: NDArray[np.int32]


@dataclass(frozen=True)
class BinnedSpatialCorrelationStatistics:
    """基線binごとの正規化相関分布とbin構成を保持する。

    Attributes:
        mean: bin内pair相関の平均。shapeは`[n_direction,n_bin,n_group]`。
        median: bin内pair相関の中央値。shapeは`[n_direction,n_bin,n_group]`。
        percentile_95: bin内pair相関の95 percentile。shapeは`[n_direction,n_bin,n_group]`。
        standard_deviation: bin内pair相関の標準偏差。shapeは`[n_direction,n_bin,n_group]`。
        interquartile_range: 75 percentileと25 percentileの差。shapeは`[n_direction,n_bin,n_group]`。
        pair_count: 各周波数・groupのpair数。shapeは`[n_bin,n_group]`。
        value_minimum: group内の基線値最小。shapeは`[n_bin,n_group]`。
        value_maximum: group内の基線値最大。shapeは`[n_bin,n_group]`。
        value_representative: group内の基線値平均。shapeは`[n_bin,n_group]`。
        group_edges: group境界。shapeは`[n_group+1]`。
    """

    mean: NDArray[np.float32]
    median: NDArray[np.float32]
    percentile_95: NDArray[np.float32]
    standard_deviation: NDArray[np.float32]
    interquartile_range: NDArray[np.float32]
    pair_count: NDArray[np.int32]
    value_minimum: NDArray[np.float32]
    value_maximum: NDArray[np.float32]
    value_representative: NDArray[np.float32]
    group_edges: NDArray[np.float32]


@dataclass(frozen=True)
class PairCompositionSpatialCorrelationStatistics:
    """中央・外側チャネルのpair構成別相関統計を保持する。

    Attributes:
        group_names: `central_central`、`central_outer`、`outer_outer`の順序。
        mean: 構成別平均。shapeは`[3,n_direction,n_bin]`。
        median: 構成別中央値。shapeは`[3,n_direction,n_bin]`。
        percentile_95: 構成別95 percentile。shapeは`[3,n_direction,n_bin]`。
        standard_deviation: 構成別標準偏差。shapeは`[3,n_direction,n_bin]`。
        interquartile_range: 構成別四分位範囲。shapeは`[3,n_direction,n_bin]`。
        pair_count: 構成別pair数。shapeは`[3]`。
    """

    group_names: tuple[str, str, str]
    mean: NDArray[np.float32]
    median: NDArray[np.float32]
    percentile_95: NDArray[np.float32]
    standard_deviation: NDArray[np.float32]
    interquartile_range: NDArray[np.float32]
    pair_count: NDArray[np.int32]


@dataclass(frozen=True)
class SparseArraySpatialCorrelationStatistics:
    """非等間隔アレイ向けの全pair・基線・pair構成別統計を保持する。

    Attributes:
        global_statistics: 全下三角pairの統計。baseline index別値は使用しない。
        physical_baseline: 物理基線長[m]でbin分けした統計。
        wavelength_normalized_baseline: `d*f/c`でbin分けした統計。
        pair_composition: 中央・外側のpair構成別統計。
        pair_distance_m: 下三角pairの座標距離。shapeは`[n_pair]`、単位はm。
    """

    global_statistics: SpatialCorrelationStatistics
    physical_baseline: BinnedSpatialCorrelationStatistics
    wavelength_normalized_baseline: BinnedSpatialCorrelationStatistics
    pair_composition: PairCompositionSpatialCorrelationStatistics
    pair_distance_m: NDArray[np.float32]


def calculate_spatial_correlation_statistics(
    direction_covariance: NDArray[Any],
    *,
    denominator_floor: float = 1.0e-20,
) -> SpatialCorrelationStatistics:
    """共分散の下三角全pairから正規化相関統計を計算する。

    Args:
        direction_covariance: 方位別共分散。shapeは`[n_direction,n_ch,n_ch,n_bin]`。
        denominator_floor: `Rii*Rjj`の最小許容power二乗。これ以下は相関0とする。

    Returns:
        最大、平均、中央値、95 percentile、および基線index別平均。
        全統計は無次元比であり、物理範囲は`[0,1]`。

    Raises:
        ValueError: 共分散shape、channel数、対角power、または安定化値が不正な場合。

    境界条件:
        自己相関1を統計へ混入させないため対角成分を除外し、下三角`i>j`だけを使う。
        複素相関を直接平均せず、各pairで`|Rij|/sqrt(Rii*Rjj)`を求めてから集約する。
        分母がfloor以下の無power pairは、数値雑音を高相関と誤認しないよう0とする。
    """

    covariance = np.asarray(direction_covariance, dtype=np.complex64)
    floor_value = float(denominator_floor)
    require_positive_float("denominator_floor", floor_value)
    require(covariance.ndim == 4, "direction_covariance must have shape (n_direction, n_ch, n_ch, n_bin).")
    require(covariance.shape[1] == covariance.shape[2], "covariance channel axes must be square.")
    require(covariance.shape[1] >= 2, "off-diagonal correlation requires at least two channels.")

    # diagonal_power shapeは`[n_direction,n_ch,n_bin]`。Hermitian共分散の対角は実powerである。
    diagonal_power = np.real(np.diagonal(covariance, axis1=1, axis2=2)).transpose(0, 2, 1)
    require(bool(np.all(diagonal_power >= -np.finfo(np.float32).eps)), "covariance diagonal power must be non-negative.")
    diagonal_power = np.maximum(diagonal_power, np.float32(0.0))

    # 下三角pair軸を明示的に作り、correlation shapeを`[n_direction,n_pair,n_bin]`へ揃える。
    first_channels, second_channels = np.tril_indices(covariance.shape[1], k=-1)
    denominator_power = diagonal_power[:, first_channels, :] * diagonal_power[:, second_channels, :]
    valid = denominator_power > np.float32(floor_value)
    pair_correlation = np.zeros(denominator_power.shape, dtype=np.float32)
    cross_magnitude = np.abs(covariance[:, first_channels, second_channels, :])
    pair_correlation[valid] = np.asarray(
        cross_magnitude[valid] / np.sqrt(denominator_power[valid]),
        dtype=np.float32,
    )
    require(bool(np.all(pair_correlation <= 1.0 + 1.0e-4)), "normalized correlation exceeds its physical range.")
    pair_correlation = np.clip(pair_correlation, 0.0, 1.0).astype(np.float32, copy=False)

    maximum = np.asarray(np.max(pair_correlation, axis=1), dtype=np.float32)
    mean = np.asarray(np.mean(pair_correlation, axis=1), dtype=np.float32)
    median = np.asarray(np.median(pair_correlation, axis=1), dtype=np.float32)
    percentile_95 = np.asarray(np.percentile(pair_correlation, 95.0, axis=1), dtype=np.float32)

    baseline_index = np.arange(1, covariance.shape[1], dtype=np.int32)
    baseline_mean = np.empty(
        (covariance.shape[0], covariance.shape[3], baseline_index.size),
        dtype=np.float32,
    )
    pair_baseline_index = first_channels - second_channels
    for output_index, channel_separation in enumerate(baseline_index):
        # 等間隔ULAでは`abs(i-j)`が同じpairは同じ物理基線長を持つため、そのpair軸だけを平均する。
        selected_pairs = pair_baseline_index == channel_separation
        baseline_mean[:, :, output_index] = np.asarray(
            np.mean(pair_correlation[:, selected_pairs, :], axis=1),
            dtype=np.float32,
        )

    return SpatialCorrelationStatistics(
        maximum=maximum,
        mean=mean,
        median=median,
        percentile_95=percentile_95,
        baseline_mean=baseline_mean,
        baseline_index=baseline_index,
    )


def _calculate_pair_correlation(
    covariance: NDArray[np.complex64],
    denominator_floor: float,
) -> tuple[NDArray[np.float32], NDArray[np.int64], NDArray[np.int64]]:
    """下三角pairの絶対正規化相関とchannel indexを返す。"""

    diagonal_power = np.real(np.diagonal(covariance, axis1=1, axis2=2)).transpose(0, 2, 1)
    require(bool(np.all(diagonal_power >= -np.finfo(np.float32).eps)), "covariance diagonal power must be non-negative.")
    diagonal_power = np.maximum(diagonal_power, np.float32(0.0))
    first_channels, second_channels = np.tril_indices(covariance.shape[1], k=-1)
    denominator_power = diagonal_power[:, first_channels, :] * diagonal_power[:, second_channels, :]
    valid = denominator_power > np.float32(denominator_floor)
    pair_correlation = np.zeros(denominator_power.shape, dtype=np.float32)
    cross_magnitude = np.abs(covariance[:, first_channels, second_channels, :])
    pair_correlation[valid] = np.asarray(
        cross_magnitude[valid] / np.sqrt(denominator_power[valid]),
        dtype=np.float32,
    )
    require(bool(np.all(pair_correlation <= 1.0 + 1.0e-4)), "normalized correlation exceeds its physical range.")
    return (
        np.clip(pair_correlation, 0.0, 1.0).astype(np.float32, copy=False),
        np.asarray(first_channels, dtype=np.int64),
        np.asarray(second_channels, dtype=np.int64),
    )


def _calculate_binned_statistics(
    pair_correlation: NDArray[np.float32],
    group_index_by_frequency_and_pair: NDArray[np.int32],
    grouping_value_by_frequency_and_pair: NDArray[np.float32],
    group_edges: NDArray[np.float32],
) -> BinnedSpatialCorrelationStatistics:
    """周波数ごとのpair group割当から相関分布とbin構成を計算する。"""

    n_direction, n_pair, n_bin = pair_correlation.shape
    n_group = int(group_edges.size - 1)
    require(group_index_by_frequency_and_pair.shape == (n_bin, n_pair), "group index must have shape (n_bin, n_pair).")
    require(grouping_value_by_frequency_and_pair.shape == (n_bin, n_pair), "grouping value must have shape (n_bin, n_pair).")
    output_shape = (n_direction, n_bin, n_group)
    mean = np.full(output_shape, np.nan, dtype=np.float32)
    median = np.full(output_shape, np.nan, dtype=np.float32)
    percentile_95 = np.full(output_shape, np.nan, dtype=np.float32)
    standard_deviation = np.full(output_shape, np.nan, dtype=np.float32)
    interquartile_range = np.full(output_shape, np.nan, dtype=np.float32)
    pair_count = np.zeros((n_bin, n_group), dtype=np.int32)
    value_minimum = np.full((n_bin, n_group), np.nan, dtype=np.float32)
    value_maximum = np.full((n_bin, n_group), np.nan, dtype=np.float32)
    value_representative = np.full((n_bin, n_group), np.nan, dtype=np.float32)

    for frequency_index in range(n_bin):
        for group_index in range(n_group):
            selected_pairs = group_index_by_frequency_and_pair[frequency_index] == group_index
            selected_count = int(np.count_nonzero(selected_pairs))
            pair_count[frequency_index, group_index] = selected_count
            if selected_count == 0:
                # pairが存在しないbinを0で埋めると低相関と誤解するため、欠測値NaNを保持する。
                continue
            selected_values = grouping_value_by_frequency_and_pair[frequency_index, selected_pairs]
            value_minimum[frequency_index, group_index] = np.min(selected_values)
            value_maximum[frequency_index, group_index] = np.max(selected_values)
            value_representative[frequency_index, group_index] = np.mean(selected_values)
            # selected correlation shapeは`[n_direction,n_selected_pair]`。pair axis=1を統計集約する。
            selected_correlation = pair_correlation[:, selected_pairs, frequency_index]
            mean[:, frequency_index, group_index] = np.mean(selected_correlation, axis=1)
            median[:, frequency_index, group_index] = np.median(selected_correlation, axis=1)
            percentile_95[:, frequency_index, group_index] = np.percentile(selected_correlation, 95.0, axis=1)
            standard_deviation[:, frequency_index, group_index] = np.std(selected_correlation, axis=1)
            quartiles = np.percentile(selected_correlation, [25.0, 75.0], axis=1)
            interquartile_range[:, frequency_index, group_index] = quartiles[1] - quartiles[0]

    return BinnedSpatialCorrelationStatistics(
        mean=mean,
        median=median,
        percentile_95=percentile_95,
        standard_deviation=standard_deviation,
        interquartile_range=interquartile_range,
        pair_count=pair_count,
        value_minimum=value_minimum,
        value_maximum=value_maximum,
        value_representative=value_representative,
        group_edges=group_edges.copy(),
    )


def _calculate_global_statistics_from_pairs(
    pair_correlation: NDArray[np.float32],
    first_channels: NDArray[np.int64],
    second_channels: NDArray[np.int64],
    n_ch: int,
) -> SpatialCorrelationStatistics:
    """既に正規化したpair相関から全pair統計を追加メモリなしで作る。"""

    baseline_index = np.arange(1, n_ch, dtype=np.int32)
    baseline_mean = np.empty(
        (pair_correlation.shape[0], pair_correlation.shape[2], baseline_index.size),
        dtype=np.float32,
    )
    pair_index_separation = first_channels - second_channels
    for output_index, channel_separation in enumerate(baseline_index):
        # このindex差別値は等間隔互換用であり、非等間隔評価の物理基線結果には使用しない。
        selected_pairs = pair_index_separation == channel_separation
        baseline_mean[:, :, output_index] = np.mean(pair_correlation[:, selected_pairs, :], axis=1)
    return SpatialCorrelationStatistics(
        maximum=np.asarray(np.max(pair_correlation, axis=1), dtype=np.float32),
        mean=np.asarray(np.mean(pair_correlation, axis=1), dtype=np.float32),
        median=np.asarray(np.median(pair_correlation, axis=1), dtype=np.float32),
        percentile_95=np.asarray(np.percentile(pair_correlation, 95.0, axis=1), dtype=np.float32),
        baseline_mean=baseline_mean,
        baseline_index=baseline_index,
    )


def calculate_sparse_array_spatial_correlation_statistics(
    direction_covariance: NDArray[Any],
    sensor_positions_m: NDArray[Any],
    frequency_hz: NDArray[Any],
    central_channel_mask: NDArray[Any],
    *,
    sound_speed_m_s: float,
    physical_baseline_edges_m: NDArray[Any],
    wavelength_normalized_edges: NDArray[Any],
    denominator_floor: float = 1.0e-20,
) -> SparseArraySpatialCorrelationStatistics:
    """非等間隔アレイの座標距離から相関統計を基線・pair構成別に計算する。

    Args:
        direction_covariance: 方位別共分散。shapeは`[n_direction,n_ch,n_ch,n_bin]`。
        sensor_positions_m: 受波器座標。shapeは`[n_ch,3]`、単位はm。
        frequency_hz: 周波数軸。shapeは`[n_bin]`、単位はHz。
        central_channel_mask: 中央密配置を示すmask。shapeは`[n_ch]`。
        sound_speed_m_s: 音速。単位はm/s。
        physical_baseline_edges_m: 物理基線bin境界。shapeは`[n_physical_group+1]`、単位はm。
        wavelength_normalized_edges: `d*f/c`のbin境界。shapeは`[n_normalized_group+1]`。
        denominator_floor: 正規化分母power二乗の安定化下限。

    Returns:
        全pair、物理基線、波長正規化基線、中央/外側pair構成別の統計。

    Raises:
        ValueError: 配列shape、座標、周波数、mask、bin境界、音速が不正な場合。

    Notes:
        非等間隔アレイではchannel index差を距離とみなさず、必ず
        `d_ij=norm(p_i-p_j)`を用いる。正規化基線は周波数ごとに`d_ij*f_k/c`とする。
    """

    covariance = np.asarray(direction_covariance, dtype=np.complex64)
    positions = np.asarray(sensor_positions_m, dtype=np.float32)
    frequencies = np.asarray(frequency_hz, dtype=np.float32)
    central_mask = np.asarray(central_channel_mask, dtype=bool)
    physical_edges = np.asarray(physical_baseline_edges_m, dtype=np.float32)
    normalized_edges = np.asarray(wavelength_normalized_edges, dtype=np.float32)
    sound_speed = float(sound_speed_m_s)
    floor_value = float(denominator_floor)
    require_positive_float("sound_speed_m_s", sound_speed)
    require_positive_float("denominator_floor", floor_value)
    require(covariance.ndim == 4, "direction_covariance must have shape (n_direction, n_ch, n_ch, n_bin).")
    require(covariance.shape[1] == covariance.shape[2], "covariance channel axes must be square.")
    require(positions.shape == (covariance.shape[1], 3), "sensor_positions_m must have shape (n_ch, 3).")
    require(frequencies.shape == (covariance.shape[3],), "frequency_hz must match n_bin.")
    require(central_mask.shape == (covariance.shape[1],), "central_channel_mask must match n_ch.")
    require(bool(np.any(central_mask)) and bool(np.any(~central_mask)), "central and outer channels must both exist.")
    require(physical_edges.ndim == 1 and physical_edges.size >= 2, "physical baseline edges must be a 1-D array with at least two values.")
    require(normalized_edges.ndim == 1 and normalized_edges.size >= 2, "normalized baseline edges must be a 1-D array with at least two values.")
    require(bool(np.all(np.diff(physical_edges) > 0.0)), "physical baseline edges must increase.")
    require(bool(np.all(np.diff(normalized_edges) > 0.0)), "normalized baseline edges must increase.")
    require(bool(np.all(np.isfinite(positions))), "sensor_positions_m must be finite.")
    require(bool(np.all(np.isfinite(frequencies))), "frequency_hz must be finite.")
    require(bool(np.all(frequencies >= 0.0)), "frequency_hz must be non-negative.")

    pair_correlation, first_channels, second_channels = _calculate_pair_correlation(covariance, floor_value)
    pair_distance_m = np.linalg.norm(positions[first_channels] - positions[second_channels], axis=1).astype(np.float32)
    require(bool(np.all(pair_distance_m > 0.0)), "sensor positions must not contain duplicate channels.")
    require(float(physical_edges[0]) <= float(np.min(pair_distance_m)), "physical baseline edges must include the shortest pair.")
    require(float(physical_edges[-1]) >= float(np.max(pair_distance_m)), "physical baseline edges must include the longest pair.")

    # 全pair統計はpair軸=1を集約する。baseline_meanは互換結果型のためindex差別も保持するが、
    # 非等間隔評価ではphysical_baselineの座標距離別統計だけを基線結果として使用する。
    global_statistics = _calculate_global_statistics_from_pairs(
        pair_correlation,
        first_channels,
        second_channels,
        covariance.shape[1],
    )

    physical_group_index = np.digitize(pair_distance_m, physical_edges[1:-1], right=False).astype(np.int32)
    physical_group_index_by_frequency = np.broadcast_to(
        physical_group_index[np.newaxis, :],
        (frequencies.size, pair_distance_m.size),
    ).copy()
    physical_value_by_frequency = np.broadcast_to(
        pair_distance_m[np.newaxis, :],
        physical_group_index_by_frequency.shape,
    ).copy()
    physical_statistics = _calculate_binned_statistics(
        pair_correlation,
        physical_group_index_by_frequency,
        physical_value_by_frequency,
        physical_edges,
    )

    normalized_value_by_frequency = (
        frequencies[:, np.newaxis] * pair_distance_m[np.newaxis, :] / np.float32(sound_speed)
    ).astype(np.float32)
    require(float(normalized_edges[0]) <= float(np.min(normalized_value_by_frequency)), "normalized edges must include the minimum d/lambda.")
    require(float(normalized_edges[-1]) >= float(np.max(normalized_value_by_frequency)), "normalized edges must include the maximum d/lambda.")
    normalized_group_index = np.digitize(
        normalized_value_by_frequency,
        normalized_edges[1:-1],
        right=False,
    ).astype(np.int32)
    normalized_statistics = _calculate_binned_statistics(
        pair_correlation,
        normalized_group_index,
        normalized_value_by_frequency,
        normalized_edges,
    )

    first_is_central = central_mask[first_channels]
    second_is_central = central_mask[second_channels]
    composition_masks = (
        first_is_central & second_is_central,
        first_is_central ^ second_is_central,
        (~first_is_central) & (~second_is_central),
    )
    composition_shape = (3, covariance.shape[0], covariance.shape[3])
    composition_mean = np.empty(composition_shape, dtype=np.float32)
    composition_median = np.empty(composition_shape, dtype=np.float32)
    composition_percentile_95 = np.empty(composition_shape, dtype=np.float32)
    composition_standard_deviation = np.empty(composition_shape, dtype=np.float32)
    composition_interquartile_range = np.empty(composition_shape, dtype=np.float32)
    composition_pair_count = np.empty(3, dtype=np.int32)
    for composition_index, selected_pairs in enumerate(composition_masks):
        selected_correlation = pair_correlation[:, selected_pairs, :]
        composition_pair_count[composition_index] = int(np.count_nonzero(selected_pairs))
        composition_mean[composition_index] = np.mean(selected_correlation, axis=1)
        composition_median[composition_index] = np.median(selected_correlation, axis=1)
        composition_percentile_95[composition_index] = np.percentile(selected_correlation, 95.0, axis=1)
        composition_standard_deviation[composition_index] = np.std(selected_correlation, axis=1)
        composition_quartiles = np.percentile(selected_correlation, [25.0, 75.0], axis=1)
        composition_interquartile_range[composition_index] = composition_quartiles[1] - composition_quartiles[0]

    return SparseArraySpatialCorrelationStatistics(
        global_statistics=global_statistics,
        physical_baseline=physical_statistics,
        wavelength_normalized_baseline=normalized_statistics,
        pair_composition=PairCompositionSpatialCorrelationStatistics(
            group_names=("central_central", "central_outer", "outer_outer"),
            mean=composition_mean,
            median=composition_median,
            percentile_95=composition_percentile_95,
            standard_deviation=composition_standard_deviation,
            interquartile_range=composition_interquartile_range,
            pair_count=composition_pair_count,
        ),
        pair_distance_m=pair_distance_m,
    )
